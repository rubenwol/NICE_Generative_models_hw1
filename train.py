"""Training procedure for NICE.
"""

import argparse
import torch
import torchvision
from torchvision import transforms
from collections import defaultdict
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import nice


def train(flow, trainloader, optimizer, epoch, device):
    flow.train()  # set to training mode
    loss = 0
    for inputs, _ in trainloader:
        inputs = inputs.view(inputs.shape[0],
                             inputs.shape[1] * inputs.shape[2] * inputs.shape[3]) # change  shape from BxCxHxW to Bx(C*H*W)
        inputs = inputs.to(device)
        # batch_size = inputs.shape[0]
        nll = -flow(inputs).mean()
        loss += nll.item()
        optimizer.zero_grad()
        nll.backward()
        optimizer.step()
    losses = loss / (len(trainloader))
    print(f'Epoch: {epoch}, Train loss: {losses}')




def test(flow, testloader, filename, epoch, sample_shape, device):
    flow.eval()  # set to inference mode
    with torch.no_grad():
        samples = flow.sample(100).cpu()
        a, b = samples.min(), samples.max()
        samples = (samples - a) / (b - a + 1e-10)
        samples = samples.view(-1, sample_shape[0], sample_shape[1], sample_shape[2])
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     './samples/' + filename + 'epoch%d.png' % epoch)

        loss = 0
        for inputs, _ in testloader:
            inputs = inputs.view(inputs.shape[0],
                                 inputs.shape[1] * inputs.shape[2] * inputs.shape[
                                     3])  # change  shape from BxCxHxW to Bx(C*H*W)
            inputs = inputs.to(device)
            nll = -flow(inputs).mean()
            loss += nll

        print(f'Test loss: {loss/len(testloader)}')

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_shape = [1, 28, 28]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.,)),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1. / 256.))  # dequantization
    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
                                              train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
                                                     train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                    train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    model_save_filename = '%s_' % args.dataset \
                          + 'batch%d_' % args.batch_size \
                          + 'coupling%d_' % args.coupling \
                          + 'coupling_type%s_' % args.coupling_type \
                          + 'mid%d_' % args.mid_dim \
                          + 'hidden%d_' % args.hidden \
                          + '.pt'
    
    filename = '%s_' % args.dataset \
                          + 'batch%d_' % args.batch_size \
                          + 'coupling%d_' % args.coupling \
                          + 'coupling_type%s_' % args.coupling_type \
                          + 'mid%d_' % args.mid_dim \
                          + 'hidden%d_' % args.hidden \
                          + 'lr%f_' % args.lr \
                          + 'prior%s_' % args.prior

    full_dim = trainset[0][0].view(-1).shape[0]

    flow = nice.NICE(
        prior=args.prior,
        coupling=args.coupling,
        coupling_type=args.coupling_type,
        in_out_dim=full_dim,
        mid_dim=args.mid_dim,
        hidden=args.hidden,
        device=device).to(device)

    optimizer = torch.optim.Adam(
        flow.parameters(), lr=args.lr)

    for epoch in tqdm(range(args.epochs)):
        train(flow, trainloader, optimizer, epoch, device)

    test(flow, testloader, filename, epoch, sample_shape, device)









if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--prior',
                        help='latent distribution.',
                        type=str,
                        default='logistic')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--coupling-type',
                        help='.',
                        type=str,
                        default='additive')
    parser.add_argument('--coupling',
                        help='.',
                        # type=int,
                        default=4)
    parser.add_argument('--mid-dim',
                        help='.',
                        type=int,
                        default=1000)
    parser.add_argument('--hidden',
                        help='.',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)
