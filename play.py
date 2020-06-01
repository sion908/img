def play(img):
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np

    import chainer
    import chainer.links as L
    import chainer.functions as F
    from chainer import serializers
    import def_fun


    # テスト用データを読み込む
    with open('test.pickle', mode='rb') as fi1:
        test = pickle.load(fi1)

    # ネットワークのインスタンスを作る
    infer_net = def_fun.MLP()

    # ネットワークに学習済みのパラメータを読み込む
    serializers.load_npz('my_mnist.model', infer_net)

    # データの準備
    

    #形の変換
    ans=[]
    for x in img:
        for y in x:
            g = 1 - sum(y) / 3 / 255
            ans.append(g)
            
    x = np.array([ans],dtype=np.float32)

    # ネットワークと同じデバイス上にデータを送る
    x = infer_net.xp.asarray(x)

    # モデルのforward関数に渡す
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y = infer_net(x)

    # Variable形式で出てくるので中身を取り出す
    ans = y.array.argmax(axis=1)
    
    return int(ans[0])

if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    
    filepath = 'img3.png'
    # img = Image
    img = np.array(Image.open(filepath).resize((28,28)))
    result = play(img)
    print(result)
    print(type(result))
    print(result.shape)

