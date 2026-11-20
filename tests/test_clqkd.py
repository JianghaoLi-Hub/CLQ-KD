"""CLQ-KD numerical/entry tests, independent of downloads or training runs."""
import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F
from ultralytics.distillation.options import resolve_clqkd, DEFAULTS
from ultralytics.distillation import method_config, trainer_for
from ultralytics.distillation.clqkd.core import filtered_loss


class CLQKDTests(unittest.TestCase):
    def test_options(self):
        _, s = resolve_clqkd('cwd', dict(clqkd=True, clqkd_alpha=3.0))
        self.assertEqual(s['clqkd_alpha'], 3.)
        self.assertEqual(s['clqkd_beta'], 1.)
        self.assertEqual(resolve_clqkd('cwd', dict(clqkd=True, clqkd_beta=3.))[1]['clqkd_beta'], 3.)
        self.assertEqual(resolve_clqkd('clqkd', {})[1]['clqkd_beta'], 0.)
        for method, kw in [('cwd', {'clqkd_beta': 1}), ('baseline', {'clqkd': True}),
                           ('clqkd', {'clqkd': True}), ('cwd', {'clqkd_alpha': 3}),
                           ('clqkd', {'clqkd_alpha': float('nan')}),
                           ('clqkd', {'clqkd_student_temperature': 0}),
                           ('clqkd', {'clqkd_prob_threshold': 1.1})]:
            with self.assertRaises(ValueError): resolve_clqkd(method, kw)
        for value in (-1., float('inf'), float('nan'), True):
            with self.assertRaises(ValueError):
                resolve_clqkd('cwd', dict(clqkd=True, clqkd_beta=value))
        for fn in (method_config, trainer_for):
            with self.assertRaises(ValueError): fn('gptkd')

    def test_combination_beta_yaml_and_explicit_priority(self):
        config='clqkd: {clqkd_beta: 0.0}\nclqkd_combination: {clqkd_beta: 2.0}\n'
        with patch('pathlib.Path.read_text', return_value=config):
            self.assertEqual(resolve_clqkd('cwd', {'clqkd':True})[1]['clqkd_beta'], 2.)
            self.assertEqual(resolve_clqkd('cwd', {'clqkd':True,'clqkd_beta':3.})[1]['clqkd_beta'], 3.)
            self.assertEqual(resolve_clqkd('clqkd', {})[1]['clqkd_beta'], 0.)

    def test_all_combination_objectives_beta_gradients(self):
        from types import SimpleNamespace
        for method in ('cwd','fgd','pkd','mgd','crosskd','gkd','ost','reviewkd','csakd'):
            for beta in (0.,1.,3.):
                with self.subTest(method=method,beta=beta):
                    # Native parts already include cls gain and batch scaling.
                    parts=torch.tensor([2.,3.,5.],requires_grad=True)
                    other=torch.tensor(7.,requires_grad=True)
                    cross=torch.tensor(11.,requires_grad=True)
                    filtered=torch.tensor(13.,requires_grad=True)
                    trainer=object.__new__(trainer_for(method))
                    trainer.model=SimpleNamespace(criterion=SimpleNamespace(clqkd_native_loss=parts))
                    trainer.clqkd_active=True
                    trainer.clqkd_settings={'clqkd_beta':beta,'clqkd_alpha':5.}
                    trainer.kd_weight=trainer.mgd_alpha=2.
                    trainer.csakd_alpha=3.;trainer.csakd_beta=4.
                    if method=='csakd':
                        trainer.csakd=SimpleNamespace()
                        trainer.compute_loss_components=lambda b:(parts.sum(),parts.detach()/2,other,cross,{})
                    else:
                        trainer.compute_loss_components=lambda b:(parts.sum(),parts.detach()/2,other,{})
                    trainer.prepare_clqkd=lambda:None
                    trainer.clear_clqkd=lambda:None
                    trainer.capture_clqkd_features=lambda x:None
                    trainer.clqkd_extra=lambda b:(filtered,5.*filtered,{})
                    trainer.record_clqkd=lambda *args:None
                    result,items=trainer.compute_training_loss({'img':torch.zeros(2,3,1,1)})
                    factor=2. if method=='csakd' else 1.
                    expected=factor*(parts[0]+beta*parts[1]+parts[2])+5.*filtered
                    expected=expected+3.*other+4.*cross if method=='csakd' else expected+4.*other
                    torch.testing.assert_close(result,expected,rtol=0,atol=0)
                    grads=torch.autograd.grad(result,(parts,other,filtered))
                    torch.testing.assert_close(grads[0],factor*torch.tensor([1.,beta,1.]),rtol=0,atol=0)
                    self.assertEqual(grads[1].item(),3. if method=='csakd' else 4.)
                    self.assertEqual(grads[2].item(),5.)
                    torch.testing.assert_close(items,torch.tensor([1.,1.5*beta,2.5]),rtol=0,atol=0)
                    if beta==1.:
                        native=parts.sum();original_items=parts.detach()
                        a,b=trainer.apply_clqkd_beta(native,original_items)
                        self.assertIs(a,native);self.assertIs(b,original_items)

    def test_formula_and_gradient(self):
        for b, a, nc in [(2, 12, 20), (1, 7, 12), (3, 5, 80)]:
            for fg_value in (False, True):
                torch.manual_seed(17)
                z = torch.randn(b,a,nc,requires_grad=True)
                t = torch.randn_like(z, requires_grad=True)
                fg = torch.full((b,a), fg_value)
                labels = torch.zeros(b,a,dtype=torch.long)
                gt = torch.tensor([0.,0.,10.,10.]).expand(b,a,4).clone()
                boxes = gt.clone(); boxes[:,::2] += 100
                # An FG position may have zero TAL quality: fg still controls masking.
                actual, counts = filtered_loss(z,(t,boxes),(fg,labels,gt),DEFAULTS)
                q = 1/(1+torch.exp(-t.detach()/1.5))
                mask = torch.ones_like(z)
                if fg_value:
                    mask[:,::2,0] = 0
                expected = (F.binary_cross_entropy(1/(1+torch.exp(-z)),q,reduction='none')*mask).sum()/a
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                ga = torch.autograd.grad(actual,z,retain_graph=True)[0]
                ge = torch.autograd.grad(expected,z)[0]
                torch.testing.assert_close(ga,ge,rtol=0,atol=0)
                self.assertEqual(int(ga[mask==0].count_nonzero()),0)
                self.assertIsNone(t.grad)
                self.assertEqual(counts['foreground'],b*a if fg_value else 0)

    def test_probability_and_iou_boundaries(self):
        # exp/log rounding differs by platform: test strictness at the exact computed probability.
        t = torch.tensor([[[-6.9067545]]]); p = float(1/(1+torch.exp(-t)))
        z = torch.zeros_like(t,requires_grad=True)
        gt = torch.tensor([[[0.,0.,10.,10.]]]); box = torch.tensor([[[0.,0.,3.,10.]]])
        ctx = (torch.ones(1,1,dtype=torch.bool),torch.zeros(1,1,dtype=torch.long),gt)
        settings = dict(DEFAULTS,clqkd_prob_threshold=p)
        _, counts = filtered_loss(z,(t,box),ctx,settings)
        self.assertEqual(counts['ignored'],1)
        settings['clqkd_prob_threshold'] = p/2
        _, counts = filtered_loss(z,(t,box),ctx,settings)
        self.assertEqual(counts['kept'],1)  # FP32 ordinary IoU exactly 0.3
        settings['clqkd_iou_threshold'] = .30001
        _, counts = filtered_loss(z,(t,box),ctx,settings)
        self.assertEqual(counts['ignored'],1)

    def test_context_required(self):
        with self.assertRaises(RuntimeError): filtered_loss(None,None,None,DEFAULTS)

    def test_same_tal_pixel_context_native_box_and_validation(self):
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg
        from ultralytics.utils.loss import v8DetectionLoss
        model = YOLO('configs/yolo11n.yaml').model.float().train()
        model.args = get_cfg()
        criterion = v8DetectionLoss(model)
        for has_gt in (False, True):
            batch = dict(img=torch.rand(1,3,64,64), batch_idx=torch.tensor([0.] if has_gt else []),
                         cls=torch.tensor([[0.]] if has_gt else []).reshape(-1,1),
                         bboxes=torch.tensor([[.5,.5,.6,.6]] if has_gt else []).reshape(-1,4))
            predictions = model(batch['img'])
            criterion.capture_clqkd = True
            native, _ = criterion(predictions,batch)
            scores, ctx = criterion.clqkd_context
            pixel_gt = ctx[2].clone()
            if has_gt and ctx[0].any():
                self.assertGreater(float(pixel_gt[ctx[0]].max()),1.)
            teacher = (torch.randn_like(scores),pixel_gt.clone())
            raw,_ = filtered_loss(scores,teacher,ctx,DEFAULTS)
            criterion.clqkd_teacher = teacher
            criterion.clqkd_settings = dict(DEFAULTS,clqkd_beta=3.)
            actual, _ = criterion(predictions,batch)
            torch.testing.assert_close(actual[[0,2]],native[[0,2]],rtol=0,atol=0)
            torch.testing.assert_close(actual[1],5*raw*model.args.cls+3*native[1])
            criterion.clqkd_teacher = None
            diagnostic,_ = criterion(predictions,batch)
            torch.testing.assert_close(diagnostic,native,rtol=0,atol=0)


if __name__ == '__main__': unittest.main()
