from block2 import block2


def stack2(x, filters, blocks, stride1=2, name=None):
    x = block2(x, filters, conv_shortcut=True, name=name+'_block1')
    for i in range(2, blocks):
        x = block2(x, filters, name=name+'_block'+str(i))
    x = block2(x, filters, stride=stride1, name=name+'_block'+str(blocks))
    return x
