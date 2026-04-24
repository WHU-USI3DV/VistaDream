import argparse
from pathlib import Path
from PIL import Image
from ops.legos.llava import Llava


def parse_args():
	parser = argparse.ArgumentParser(
		description='Describe one image with LLaVA only.'
	)
	parser.add_argument(
		'--image',
		type=str,
		required=True,
		help='Path to input image.',
	)
	parser.add_argument(
		'--ckpt',
		type=str,
		default='llava-hf/bakLlava-v1-hf',
		help='LLaVA model path or Hugging Face id.',
	)
	parser.add_argument(
		'--device',
		type=str,
		default='cuda',
		choices=['cuda', 'cpu'],
		help='Inference device.',
	)
	parser.add_argument(
		'--offline',
		action='store_true',
		help='Load from local cache/path only (no network).',
	)
	parser.add_argument(
		'--prompt',
		type=str,
		default=(
			'<image>\n '
			'USER: Detaily imagine and describe the scene this image taken from? '
			'\n ASSISTANT: This image is taken from a scene of '
		),
		help='Custom LLaVA prompt.',
	)
	parser.add_argument(
		'--output',
		type=str,
		default='',
		help='Optional txt output file. If empty, only print to stdout.',
	)
	return parser.parse_args()


def main():
	args = parse_args()
	image_path = Path(args.image)
	if not image_path.exists():
		raise FileNotFoundError(f'Image not found: {image_path}')

	image = Image.open(image_path).convert('RGB')

	# Try requested device first, then gracefully fallback to CPU.
	try:
		tool = Llava(
			device=args.device,
			llava_ckpt=args.ckpt,
			offline=args.offline,
		)
	except Exception as e:
		if args.device == 'cuda':
			print(f'[WARN] Failed to load LLaVA on cuda: {e}')
			print('[INFO] Fallback to cpu.')
			tool = Llava(
				device='cpu',
				llava_ckpt=args.ckpt,
				offline=args.offline,
			)
		else:
			raise

	answer = tool(image, args.prompt)
	print(answer[len(args.prompt):].strip())

	if args.output:
		out_path = Path(args.output)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		out_path.write_text(answer + '\n', encoding='utf-8')
		print(f'[INFO] Saved description to: {out_path}')


if __name__ == '__main__':
	main()

