import tensorflow as tf
import numpy as np
from model import GAN
from utils import *
import argparse
import pickle
import scipy.misc as misc
from PIL import Image 
import time

def test(args):
    model = GAN(
        batch_size=args.bs,  
        noise_dim=args.noise_dim, 
        learning_rate=args.lr, 
        trainable=True)
    
    model.build()

    saver = tf.train.Saver(max_to_keep=20)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config) 

    latest_ckpt = tf.train.latest_checkpoint(args.loadpath)
    saver.restore(sess, latest_ckpt)
    print('restore from', latest_ckpt) 
 
    batch_y_tot = gen_testdata()
    # batch_y_tot = gen_fromfile(args.testfile) 
    new_im = Image.new('RGB', (64*5,64*len(batch_y_tot))) 
    
    for j in range(len(batch_y_tot)):
         
        batch_y = np.tile(batch_y_tot[j], (5,1))
        noise = np.random.uniform(-1, 1, [batch_y.shape[0], args.noise_dim]) 
        generated_test = sess.run(model.sampler, feed_dict={model.noises: noise, model.labels: batch_y})

        for i in range(5): 
            generated = (generated_test[i] + 1) * 127.5 # scale from [-1., 1.] to [0., 255.]
            generated = np.clip(generated, 0., 255.).astype(np.uint8)
            generated = misc.imresize(generated, [64, 64, 3]) 

            gen_path = 'samples/sample_' + str(j+1) + '_' + str(i+1) + '.jpg'
            misc.imsave(gen_path, generated)

            new_im.paste(Image.fromarray(generated, "RGB"), (64*i,64*j))

        path = 'samples_' + str(j) 
        pickle.dump(noise, open(path, 'wb'))

    gen_path = 'samples/' + '1' + '.jpg'
    new_im.save(gen_path) 
    print('gen results:', len(batch_y_tot), '* 5 in samples')


def train(dataset, args, retrain):
    model = GAN(
        batch_size=args.bs, 
        noise_dim=args.noise_dim, 
        learning_rate=args.lr, 
        trainable=True)
    
    model.build()

    saver = tf.train.Saver(max_to_keep=20)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    if retrain == True:
        latest_ckpt = tf.train.latest_checkpoint(args.loadpath)
        saver.restore(sess, latest_ckpt)
        print('restore from', latest_ckpt) 

    print ('Start training, method=%s, lr=%f, batch_size=%d, epoch=%d, comment=%s'% (model.name, args.lr, args.bs, args.ep, args.comment))

    path_args = args.savepath + 'output.log'
    fwirte = open(path_args, 'w')

    for ep in range(1, args.ep+1):
        d_tot_loss = []
        g_tot_loss = []
        total_loss = 0

        start_time = time.time()

        for step in range(0, dataset.iters):
            for d_train in range(5):
                batch_x, wrong_x, batch_y, wrong_y = dataset.next_batch()
                noise = np.random.uniform(-1, 1, [batch_x.shape[0], args.noise_dim])
                
            # update D
                d_loss, _ = sess.run([model.d_loss_op, model.d_train_op], 
                    feed_dict={ model.imgs: batch_x, model.labels: batch_y, model.noises: noise, 
                                model.imgs_wrong: wrong_x ,model.labels_wrong:wrong_y})
                d_tot_loss.append(d_loss)

            # update G  
            noise = np.random.uniform(-1, 1, [batch_x.shape[0], args.noise_dim])

            generated, g_loss, _ = sess.run([model.generated_op, model.g_loss_op, model.g_train_op], 
                feed_dict={ model.imgs: batch_x, model.labels: batch_y, model.noises: noise, 
                                model.imgs_wrong: wrong_x ,model.labels_wrong:wrong_y}) 
            g_tot_loss.append(g_loss)
          
            if step % 2 == 0 or step == 1:
                print("Epoch: %5d, Step: %4d/%4d, D_loss: %.4f, G_loss: %.4f          " % 
                    (ep, step, dataset.iters, np.mean(d_tot_loss), np.mean(g_tot_loss)), end='\r')

        end_time = time.time() 

        fwirte.write("Epoch: %5d, Take time: %4.1fs, D_loss: %.4f, G_loss: %.4f\n" % 
                    (ep, (end_time-start_time), np.mean(d_tot_loss), np.mean(g_tot_loss) ))
        print("Epoch: %5d, Take time: %4.1fs, D_loss: %.4f, G_loss: %.4f          " % 
                    (ep, (end_time-start_time), np.mean(d_tot_loss), np.mean(g_tot_loss)))

        if ep % 10 == 0 or ep == 1:
            # test 
            new_im = Image.new('RGB', (64*5,64*5))
            for j in range(5):
                batch_y = gen_testdata()
                noise = np.random.uniform(-1, 1, [batch_y.shape[0], args.noise_dim])
                generated_test = sess.run(model.sampler, feed_dict={model.noises: noise, model.labels: batch_y})

                for i in range(5):
                    generated = (generated_test[i] + 1) * 127.5 # scale from [-1., 1.] to [0., 255.]
                    generated = np.clip(generated, 0., 255.).astype(np.uint8)
                    generated = misc.imresize(generated, [64, 64, 3]) 
                    new_im.paste(Image.fromarray(generated, "RGB"), (64*i,64*j))

            gen_path = args.savepath+'generated/' + str(ep) + '.jpg'
            new_im.save(gen_path) 
            saver.save(sess, args.savepath+'model.ckpt', global_step=ep)

    print('Done')
    print("Model saved in file: %s\n" % args.savepath)

    fwirte.write('Done')
    fwirte.close()

def gen_testdata():
    tag_dict = {'orange hair':0, 'white hair':1, 'aqua hair':2, 'gray hair':3, 'green hair':4, 'red hair':5, 'purple hair':6, 
            'pink hair':7, 'blue hair':8, 'black hair':9, 'brown hair':10, 'blonde hair':11, 
            'gray eyes':12, 'black eyes':13, 'orange eyes':14, 'pink eyes':15, 'yellow eyes':16,
            'aqua eyes':17, 'purple eyes':18, 'green eyes':19, 'brown eyes':20, 'red eyes':21, 'blue eyes':22}

    strs = ['blue hair, red eyes', 'black hair, blue eyes', 'red hair, green eyes']
    batch_y = []
    for t in strs:
        spl_t = t.split(', ')
        tag_onehot = np.zeros(23)
        for tag in spl_t:
            # if tag in tag_dict:
            tag_onehot[tag_dict[tag]] = 1
        batch_y.append(tag_onehot)
    return np.asarray(batch_y).astype(np.float32)

def gen_fromfile(path):
    tag_dict = {'orange hair':0, 'white hair':1, 'aqua hair':2, 'gray hair':3, 'green hair':4, 'red hair':5, 'purple hair':6, 
            'pink hair':7, 'blue hair':8, 'black hair':9, 'brown hair':10, 'blonde hair':11, 
            'gray eyes':12, 'black eyes':13, 'orange eyes':14, 'pink eyes':15, 'yellow eyes':16,
            'aqua eyes':17, 'purple eyes':18, 'green eyes':19, 'brown eyes':20, 'red eyes':21, 'blue eyes':22}

    testfile = open(path, 'r').readlines()
    print(len(testfile))
    batch_y = []
    for line in testfile:
        tag_onehot = np.zeros(23)
        tmp = line.split(',')
        ind = tmp[0]
        tags = tmp[1]
        tag_sp = tags.split()
        for i in range(len(tag_sp)):
            if tag_sp[i] == 'hair':
                t = tag_sp[i-1] + ' hair'
                tag_onehot[tag_dict[t]] = 1
            elif tag_sp[i] == 'eyes':
                t = tag_sp[i-1] + ' eyes'
                tag_onehot[tag_dict[t]] = 1
        batch_y.append(tag_onehot)
    return np.asarray(batch_y).astype(np.float32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='set this to train.')
    parser.add_argument('--retrain', action='store_true', help='set this to retrain.')
    parser.add_argument('--test', action='store_true', help='set this to testing.')
    parser.add_argument('--test_noise', action='store_true', help='set this to testing.') 
    parser.add_argument('--lr', metavar='', type=float, default=0.0001, help='learning rate.')
    parser.add_argument('--ep', metavar='', type=int, default=10000, help='number of epochs.')
    parser.add_argument('--bs', metavar='', type=int, default=64, help='batch size.')
    parser.add_argument('--noise_dim', metavar='', type=int, default=100, help='number of classes (output).')
    parser.add_argument('--savepath', metavar='', type=str, default="", help='save/load model path.') # model_rnn/model.ckpt
    parser.add_argument('--loadpath', metavar='', type=str, default="", help='save/load model path.') # model_rnn/
    parser.add_argument('--input', metavar='', type=str, default="data/", help='input data path.')
    parser.add_argument('--testfile', metavar='', type=str, default="testcase.txt", help='test file path.')
    parser.add_argument('--comment', metavar='', type=str, default="", help='describe this model')
    parser.add_argument('--output', metavar='', type=str, default="", help='the path to save output file')

    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0: raise SystemExit('Unknown argument: {}'.format(unparsed))

    if args.train or args.retrain:
        if args.savepath is None :
            print('give args --savepath yourpath')
        else:
            path_x = args.input + 'imgs_big.dat'
            path_y = args.input + 'tags_onehot.dat'

            dataset = DataLoader(path_x, path_y, args.bs)
            train(dataset, args, args.retrain)

            objarg = parser.parse_args()
            path_args = args.savepath.split('/')[0] + '/args.txt'
            with open(path_args, 'w') as output_file:
                output_file.write("%s\n" % objarg) 
    if args.test:
        if args.loadpath is None:
            print('give args --loadpath yourpath')
        else:

            test(args)

    if args.test_noise:
        if args.loadpath is None:
            print('give args --loadpath yourpath')
        else:

            test_noise(args)

