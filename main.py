# import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7,8'

# import tensorflow as tf
# print('n_gpu: {}'.format(len(tf.config.experimental.list_physical_devices('GPU'))))

import gpt_2_simple as gpt2

def finetune():
	sess = gpt2.start_tf_sess()
	gpt2.finetune(
		sess,
		dataset='skyrim_fiction_books.txt',
		model_name='355M',
		steps=4000,
		restore_from='latest',
		run_name='run3',
		print_every=5,
		sample_every=200,
		save_every=500
	)

def generate_trained(prefix):
	sess = gpt2.start_tf_sess()
	gpt2.load_gpt2(
		sess,
		run_name='run3',
	)
	gpt2.generate(
		sess,
		run_name='run3',
		length=1023,
		temperature=0.7,
		top_p=0.9,
		prefix=prefix,
		nsamples=1,
		batch_size=1
	)

def generate_pretrained(prefix):
	sess = gpt2.start_tf_sess()
	gpt2.load_gpt2(sess, model_name='774M')
	gpt2.generate(
		sess,
		model_name='774M',
		prefix=prefix,
		length=1023,
		temperature=0.7,
		top_p=0.9,
		nsamples=5,
		batch_size=5
	)

if __name__ == '__main__':
	# gpt2.download_gpt2(model_name='124M')
	# gpt2.download_gpt2(model_name='355M')
	# gpt2.download_gpt2(model_name='774M')
	# finetune()

	prefix = 'Chapter 1'
	# prefix = 'Ours is to smile at your passing, friend.'
	# prefix = 'He was a powerful foe. Bob knew little about him, but that he always seemed to be doing something strange. Because of this prejudice,'
	# prefix = 'You\'re not as dumb as you look.'
	# prefix = 'I miss you, Daisy. Although our house is cleaner than it’s ever been, it is amiss without tumbleweeds of your fur. When we meet again, I promise, we’ll still be the best buddies in town. I love you. You are the best.'
	# prefix = 'A tale of three thieves and an ill-fated heist.'
	# prefix = 'I dare not write where we stay for fear of endangering the good people of this house should this diary be discovered. We have been shown a kindness by this family once known to the'
	# prefix = 'A fine day to you, friend. May you die with a sword in your hands.'

	generate_trained(prefix)
	# generate_pretrained(prefix)
