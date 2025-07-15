# import torch.nn as nn
# import MinkowskiEngine as ME
# import torch
# #ME.MinkowskiNetwork
# class ExampleNetwork(nn.Module):

#     def __init__(self, in_feat, out_feat, D):#D
#         super(ExampleNetwork, self).__init__()
#         self.conv1 = nn.Sequential(
#             ME.MinkowskiConvolution(
#                 in_channels=in_feat,
#                 out_channels=64,
#                 kernel_size=3,
#                 stride=2,
#                 dilation=1,
#                 bias=False,
#                 dimension=D),
#             ME.MinkowskiBatchNorm(64),
#             ME.MinkowskiReLU())
#         self.conv2 = nn.Sequential(
#             ME.MinkowskiConvolution(
#                 in_channels=64,
#                 out_channels=128,
#                 kernel_size=3,
#                 stride=2,
#                 dimension=D),
#             ME.MinkowskiBatchNorm(128),
#             ME.MinkowskiReLU())
#         self.pooling = ME.MinkowskiGlobalPooling()
#         self.linear = ME.MinkowskiLinear(128, out_feat)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.pooling(out)
#         return self.linear(out)
    
# def main():
#     # Example input data
#     coordinates = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=torch.int32)
#     features = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)

#     # Create a sparse tensor
#     input_tensor = ME.SparseTensor(features=features, coordinates=coordinates)
#     # Define the network
#     print(input_tensor)
#     in_feat = 2  # 2Number of input features
#     out_feat = 10  # Number of output features
#     D = 3  # 3Dimension of the input data

#     model = ExampleNetwork(in_feat, out_feat, D)

#     # Forward pass
#     output = model(input_tensor)

#     print(output)
    
# if __name__ == '__main__':
#     main()
from MinkowskiEngine.MinkowskiOps import (
    MinkowskiToSparseTensor,
    to_sparse,
    dense_coordinates,
    MinkowskiToDenseTensor,
)
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from PIL import Image
import numpy as np
import torch.utils.checkpoint as cp
def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

class SpatialPriorModuleSparse(ME.MinkowskiNetwork):
    def __init__(self, inplanes=64, embed_dim=1024, spatial_dim=1024, in_ch=1,with_cp=False,D=2):
        super().__init__(D)
        self.with_cp = with_cp
        self.sparse = MinkowskiToSparseTensor
        self.embed_dim = embed_dim
        # self.coordinates = dense_coordinates(torch.Size([2, in_ch, spatial_dim,spatial_dim]))
        # self.sparse = MinkowskiToSparseTensor(remove_zeros=True,coordinates=self.coordinates)
        self.stem_sparse = nn.Sequential(*[
            ME.MinkowskiConvolution(in_ch, inplanes, kernel_size=3, stride=2, dilation=1, bias=False, dimension=D),
            # nn.Conv2d(in_ch, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            ME.MinkowskiBatchNorm(inplanes),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(inplanes, inplanes, kernel_size=3, stride=1, dilation=1, bias=False, dimension=D), 
            # nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            ME.MinkowskiBatchNorm(inplanes),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(inplanes, inplanes, kernel_size=3, stride=1, dilation=1, bias=False, dimension=D), 
            # nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            ME.MinkowskiBatchNorm(inplanes),
            ME.MinkowskiReLU(),
            ME.MinkowskiMaxPooling(kernel_size=3, stride=2,dimension=D)# padding=1
        ])
        self.conv2 = nn.Sequential(*[
            ME.MinkowskiConvolution(inplanes, 2 * inplanes, kernel_size=3, stride=2, dilation=1, bias=False, dimension=D),
            # nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            ME.MinkowskiBatchNorm(2 * inplanes), #Sync
            ME.MinkowskiReLU()
        ])
        self.conv3 = nn.Sequential(*[
            ME.MinkowskiConvolution(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, dilation=1, bias=False, dimension=D),
            # nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            ME.MinkowskiBatchNorm(4 * inplanes), #Sync
            ME.MinkowskiReLU()
        ])
        self.conv4 = nn.Sequential(*[
            ME.MinkowskiConvolution(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, dilation=1, bias=False, dimension=D),
            # nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            ME.MinkowskiBatchNorm(4 * inplanes),
            ME.MinkowskiReLU()
        ])
        self.fc1 = ME.MinkowskiConvolution(inplanes, embed_dim, kernel_size=1, stride=1, dilation=1, bias=True, dimension=D)
        self.fc2 = ME.MinkowskiConvolution(2*inplanes, embed_dim, kernel_size=1, stride=1, dilation=1, bias=True, dimension=D)
        self.fc3 = ME.MinkowskiConvolution(4*inplanes, embed_dim, kernel_size=1, stride=1, dilation=1, bias=True, dimension=D)
        self.fc4 = ME.MinkowskiConvolution(4*inplanes, embed_dim, kernel_size=1, stride=1, dilation=1, bias=True, dimension=D)
        # self.dense1 = MinkowskiToDenseTensor(torch.Size([2, embed_dim, spatial_dim//4,spatial_dim//4]))
        # self.dense2 = MinkowskiToDenseTensor(torch.Size([2, embed_dim, spatial_dim//8,spatial_dim//8]))
        # self.dense3 = MinkowskiToDenseTensor(torch.Size([2, embed_dim, spatial_dim//16,spatial_dim//16]))
        # self.dense4 = MinkowskiToDenseTensor(torch.Size([2, embed_dim, spatial_dim//32,spatial_dim//32]))
        self.dense = MinkowskiToDenseTensor

    def forward(self, x):
        
        def _inner_forward(x):
            bs,ch,sp,sp=x.shape
            self.coordinates = dense_coordinates(torch.Size([bs, ch, sp,sp]))
            x=self.sparse(remove_zeros=True,coordinates=self.coordinates)(x)
            # x=self.sparse(x)
            c1 = self.stem_sparse(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)
            # c1=self.dense1(c1)
            # c2=self.dense2(c2)
            # c3=self.dense3(c3)
            # c4=self.dense4(c4)
            c1=self.dense(torch.Size([2, self.embed_dim, sp//4,sp//4]))(c1)
            c2=self.dense(torch.Size([2, self.embed_dim, sp//8,sp//8]))(c2)
            c3=self.dense(torch.Size([2, self.embed_dim, sp//16,sp//16]))(c3)
            c4=self.dense(torch.Size([2, self.embed_dim, sp//32,sp//32]))(c4)
            bs, dim, _, _ = c1.shape
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s
    
            return c1, c2, c3, c4
        
        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs

# def create_input_batch(batch, is_minknet, device="cuda", quantization_size=0.05):
#     if is_minknet:
#         batch["coordinates"][:, 1:] = batch["coordinates"][:, 1:] / quantization_size
#         return ME.TensorField(
#             coordinates=batch["coordinates"],
#             features=batch["features"],
#             device=device,
#         )
#     else:
#         return batch["coordinates"].permute(0, 2, 1).to(device)
# class ExampleNetwork(ME.MinkowskiNetwork):
#     def __init__(self, in_feat, out_feat, D):
#         super(ExampleNetwork, self).__init__(D)
#         self.conv1 = nn.Sequential(
#             ME.MinkowskiConvolution(
#                 in_channels=in_feat,
#                 out_channels=64,
#                 kernel_size=3,
#                 stride=1,
#                 dilation=1,
#                 bias=False,
#                 dimension=D),
#             ME.MinkowskiBatchNorm(64),
#             ME.MinkowskiReLU())
#         self.conv2 = nn.Sequential(
#             ME.MinkowskiConvolution(
#                 in_channels=64,
#                 out_channels=128,
#                 kernel_size=3,
#                 stride=1,
#                 dimension=D),
#             ME.MinkowskiBatchNorm(128),
#             ME.MinkowskiReLU())
#         self.pooling = ME.MinkowskiGlobalPooling()
#         self.linear = ME.MinkowskiLinear(128, out_feat)

    # def forward(self, x):
    #     out = self.conv1(x)
    #     out = self.conv2(out)
    #     out = self.pooling(out)
    #     return self.linear(out)

def main():
    # Example input data
    # coordinates = torch.tensor([[[0, 0], [1, 1], [2, 2],[3, 3], [4, 4], [5, 5]]], dtype=torch.int32)
    # features = torch.tensor([[1, 2], [3, 4], [6, 7],[8,9],[10,11],[12,13]], dtype=torch.float32)

    # Create a sparse tensor
    # input_tensor = ME.SparseTensor(features, coordinates=coordinates)
    data_batch_0 = [
    [0, 0, 2.1, 0, 0],
    [0, 1, 1.4, 3, 0],
    [0, 0, 4.0, 0, 0]
    ]
    data_batch_1 = [
    [0, 0, 2.1, 0, 0],
    [0, 1, 1.4, 3, 0],
    [0, 0, 4.0, 0, 0]
    ]
    # /media/data4/sora/datasets/DELIVER_2/samples/lidar/training/cloud_MAP_1_point2_095300_lidar_front.png
    # datum = np.array(Image.open('/media/data4/sora/datasets/DELIVER_2/samples/lidar/training/cloud_MAP_1_point2_095300_lidar_front.png').resize((1024,1024)),dtype=np.float32)
    datum=np.array(Image.open('/media/data4/sora/datasets/DELIVER_2/samples/event/training/cloud_MAP_1_point2_095350_event_front.png').resize((1024,1024)),dtype=np.float32)
    # datum1 = np.array(Image.open('/media/data4/sora/datasets/DELIVER_2/samples/lidar/training/cloud_MAP_1_point2_095300_lidar_front.png').resize((1024,1024)),dtype=np.float32)
    datum1=np.array(Image.open('/media/data4/sora/datasets/DELIVER_2/samples/event/training/cloud_MAP_1_point2_095350_event_front.png').resize((1024,1024)),dtype=np.float32)
    datum_bs = np.concatenate([datum[np.newaxis,:,:],datum1[np.newaxis,:,:]],axis=0)
    # datum_bs = datum_bs[:,np.newaxis,:,:]
    datum_bs = datum_bs.transpose(0,3,2,1)
    model = SpatialPriorModuleSparse(inplanes=64, embed_dim=1024,spatial_dim=1024, in_ch=3,with_cp=False,D=2)
    # Move model to device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # .to(device)
    
    
    output = model(torch.from_numpy(datum_bs))
    
    # coordinates=dense_coordinates(torch.from_numpy(datum[np.newaxis,np.newaxis,:,:]).shape)
    # mink=MinkowskiToSparseTensor(remove_zeros=True,coordinates=coordinates)
    
    # # datum=datum.transpose(2,0,1)
    # # datum_sparsed_easy_1=to_sparse(torch.from_numpy(np.array(data_batch_0)[np.newaxis,np.newaxis,:,:]))
    # # datum_sparsed_easy=mink(torch.from_numpy(np.array(data_batch_0)[np.newaxis,np.newaxis,:,:]))
    # datum_sparsed=to_sparse(torch.from_numpy(datum[np.newaxis,np.newaxis,:,:]))
    # # coordinates=dense_coordinates(torch.from_numpy(datum[np.newaxis,np.newaxis,:,:]).shape)
    # dat=mink(torch.from_numpy(datum[np.newaxis,np.newaxis,:,:]))
    # orig=MinkowskiToDenseTensor(torch.from_numpy(datum[np.newaxis,np.newaxis,:,:]).shape)(dat)
    def to_sparse_coo(data):
        # An intuitive way to extract coordinates and features
        coords, feats = [], []
        for i, row in enumerate(data):
            for j, val in enumerate(row):
                if val != 0:
                    coords.append([i, j])
                    feats.append([val])
        return torch.IntTensor(coords), torch.FloatTensor(feats)
    # datum_sparsed = to_sparse_coo(datum)
    # to_sparse_coo(data)
    coords0, feats0 = to_sparse_coo(data_batch_0)
    coords1, feats1 = to_sparse_coo(data_batch_1)
    coords, feats = ME.utils.sparse_collate(
        [coords0, coords1], [feats0, feats1])
    input_tensor = ME.SparseTensor(feats, coordinates=coords)
    # coords0, feats0 = to_sparse_coo(data_batch_0)
    # coords1, feats1 = to_sparse_coo(data_batch_1)
    # coords, feats = ME.utils.sparse_collate(
    #     [coords0, coords1], [feats0, feats1])

    # sparse tensors
    # A = ME.SparseTensor(coordinates=coords, features=feats)
    # conv = ME.MinkowskiConvolution(
    #     in_channels=1, out_channels=2, kernel_size=3, stride=2, dimension=2)
    # B = conv(A)
    # Define the network
    # dense_tensor = torch.rand(3, 4, 11, 11, 11, 11)  # BxCxD1xD2x....xDN
    # # # dense_tensor.requires_grad = True
    # # # dummy=torch.rand(3, 4, 8,8,8,8)
    # # # Since the shape is fixed, cache the coordinates for faster inference
    # coordinates = dense_coordinates(dense_tensor.shape)
    # ten=MinkowskiToSparseTensor(coordinates=coordinates)(dense_tensor)
    # network = nn.Sequential(
    #     # Add layers that can be applied on a regular pytorch tensor
    #     # nn.ReLU(),
    #     # MinkowskiToSparseTensor(coordinates=coordinates),#coordinates=coordinates
    #     ME.MinkowskiConvolution(4, 5, stride=2, kernel_size=3, dimension=4),#4
    #     ME.MinkowskiBatchNorm(5),
    #     ME.MinkowskiReLU(),
    #     # ME.MinkowskiConvolutionTranspose(5, 6, stride=2, kernel_size=3, dimension=4),
    #     MinkowskiToDenseTensor(
    #         # dense_tensor.shape
    #         # dummy.shape
    #     ),  # must have the same tensor stride.
    # )

    # for i in range(5):
    #     print(f"Iteration: {i}")
    #     output = network(ten)
    #     # output=network(dat)

    # network2 = nn.Sequential(
    #     # Add layers that can be applied on a regular pytorch tensor
    #     # nn.ReLU(),
    #     # MinkowskiToSparseTensor(coordinates=coordinates),#coordinates=coordinates
    #     ME.MinkowskiConvolution(1, 5, stride=2, kernel_size=3, dimension=2),#4
    #     ME.MinkowskiBatchNorm(5),
    #     ME.MinkowskiReLU(),
    #     # ME.MinkowskiConvolutionTranspose(5, 6, stride=2, kernel_size=3, dimension=4),
    #     MinkowskiToDenseTensor(
    #         # dense_tensor.shape
    #         # dummy.shape
    #     ),  # must have the same tensor stride.
    # )
    # for i in range(5):
    #     print(f"Iteration: {i}")
    #     output = network2(dat)
    #     # output=network(dat)
    # conv=ME.MinkowskiConvolution(1, 5, stride=2, kernel_size=3, dimension=2)#4
    # output = conv(datum_sparsed)
    
    # in_feat = 1  # Number of input features
    # out_feat = 10  # Number of output features
    # D = 2  # Dimension of the input data
    # datum_sparsed_easy_1=to_sparse(torch.from_numpy(np.array(data_batch_0)[np.newaxis,np.newaxis,:,:]))
    
    # model = ExampleNetwork(in_feat, out_feat, D)
    # # model=SpatialPriorModuleSparse(in_feat,out_feat,D)
    # # Forward pass
    # output = model(datum_sparsed)

    print(output)

if __name__ == "__main__":
    main()