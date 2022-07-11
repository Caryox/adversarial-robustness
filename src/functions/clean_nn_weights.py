def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
	Usage: NETWORK.apply(reset_weights)
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()