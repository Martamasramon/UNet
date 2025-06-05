from runet.visualize  import RUNetVisualizer
# from runet.evaluate   import RUNetEvaluation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',   type=str,  default='checkpoints_2605_1715_stage_1_best')
parser.add_argument('--img_size',     type=int,  default=128)
parser.add_argument('--drop_first',   type=float,default=0.1)
parser.add_argument('--drop_last',    type=float,default=0.5)
parser.add_argument('--batch_size',   type=int,  default=5)
args, unparsed = parser.parse_known_args()
print('\n',args)

## VISUALISE RESULTS
checkpoint_unet = f"checkpoints/{args.checkpoint}.pth"

visualizer_runet = RUNetVisualizer(args.drop_first, args.drop_last, img_size=args.img_size)
visualizer_runet.visualize_runet(checkpoint_unet, batch_size=args.batch_size)

# ## EVALUATE
# checkpoint_unet = "checkpoints/perceptual_loss_RUNET_var_blur.pth"
# evaluator_runet = RUNetEvaluation()
# evaluator_runet.evaluate(checkpoint_unet)
