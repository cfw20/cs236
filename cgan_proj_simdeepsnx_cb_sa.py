import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn
from tqdm import tqdm
from models.interface import ConditionedGenerativeModel
from torch.nn import Parameter


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class CDCGAN_D(nn.Module):
    # initializers
    def __init__(self, embed_dim=10, n_filters=128, num_classes=10):
        super(CDCGAN_D, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.conv1 = ConvSNLRelu(3, n_filters//2, 3, 1, 1)
        self.conv2 = ConvSNLRelu(n_filters//2, n_filters, 4, 2, 1)
        self.conv3 = ConvSNLRelu(n_filters, n_filters, 3, 1, 1)
        self.conv5 = ConvSNLRelu(n_filters, n_filters*2, 4, 2, 1)
        self.conv6 = ConvSNLRelu(n_filters*2, n_filters*2, 3, 1, 1)
        self.conv7 = ConvSNLRelu(n_filters*2, n_filters*4, 4, 2, 1)
        self.conv8 = ConvSNLRelu(n_filters*4, n_filters*4, 3, 1, 1)
        self.fc1 = SpectralNorm(nn.Linear(512 * 4 * 4, num_classes))
        self.fc1.apply(init_xavier_uniform)
        
        self.attn1 = Self_Attn(n_filters, 'relu')
#        self.attn2 = Self_Attn(32, 'relu')

        
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, embed):
        x = self.conv1(input)
        x = self.conv2(x)
        x, p1 = self.attn1(x)
        x = self.conv3(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.fc1(x.view(-1, 512 * 4 * 4))
#        print('x',x.size())
#        print('embed',embed.size())
        x = torch.bmm(x.view(-1, 1, self.num_classes), embed.view(-1, self.num_classes, 1))
        x = F.sigmoid(x)

        return x

class CDCGAN_G(nn.Module):
    def __init__(self, z_dim=100, embed_dim=10, n_filters=128, num_classes=10):
        super(CDCGAN_G, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        
        self.embed_dim = embed_dim
#        self.conv_embed = DeconvBNRelu(768, self.embed_dim, 3, 1, 1)
#        self.conv_embed = nn.Conv2d(768, self.embed_dim, 3, 1, 1)
#        self.conv_embed.apply(init_xavier_uniform)

        self.deconv1_1 = DeconvBNRelu(z_dim, n_filters*4, 4, 1, 0, self.embed_dim)
#        self.deconv1_2 = DeconvBNRelu(embed_dim, n_filters*1, 4, 1, 0)
        self.deconv2 = DeconvBNRelu(n_filters*4, n_filters*2, 4, 2, 1, self.embed_dim)
        self.deconv3 = DeconvBNRelu(n_filters*2, n_filters, 4, 2, 1, self.embed_dim)
        self.deconv5 = DeconvBNRelu(n_filters, n_filters//2, 4, 2, 1, self.embed_dim)
        self.deconv4 = nn.ConvTranspose2d(n_filters//2, 3, 3, 1, 1)
        self.deconv4.apply(init_xavier_uniform)

        self.attn1 = Self_Attn(n_filters, 'relu')


    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, embed):
#        embed = F.relu(self.conv_embed(embed))
        embed = torch.squeeze(embed)
#        print('embed_squeezed', embed.size())
        embed = embed[:, :self.embed_dim]
#        print('embed_sliced', embed.size())        
        x = self.deconv1_1(input, embed)
#        print('x',x.size())
#        y = self.deconv1_2(embed)
#        print('y',y.size())
#        x = torch.cat([x, y], 1)
#        print('concat',x.size())
        x = self.deconv2(x, embed)
        x = self.deconv3(x, embed)
        x, p1 = self.attn1(x)
        x = self.deconv5(x, embed)
        x = F.tanh(self.deconv4(x))
        
        return x

    def sample(self, captions_embd):
        '''
        :param captions_embd: torch.FloatTensor bsize * embd_size
        :return: imgs : torch.FloatTensor of size n_imgs * c * h * w
        '''

        z = torch.randn(captions_embd.shape[0], self.z_dim, 1, 1).to(captions_embd.device)
        with torch.no_grad():
            gen_imgs = self.forward(z, captions_embd.unsqueeze(dim=2).unsqueeze(dim=3))
        gen_imgs = (gen_imgs + 1) / 2

        return gen_imgs
    
    
class SpectralNorm(nn.Module):
	def __init__(self, module, name='weight', power_iterations=1):
		super(SpectralNorm, self).__init__()
		self.module = module
		self.name = name
		self.power_iterations = power_iterations
		if not self._made_params():
			self._make_params()

	def _update_u_v(self):
		u = getattr(self.module, self.name + "_u")
		v = getattr(self.module, self.name + "_v")
		w = getattr(self.module, self.name + "_bar")

		height = w.data.shape[0]
		_w = w.view(height, -1)
		for _ in range(self.power_iterations):
			v = l2normalize(torch.matmul(_w.t(), u))
			u = l2normalize(torch.matmul(_w, v))

		sigma = u.dot((_w).mv(v))
		setattr(self.module, self.name, w / sigma.expand_as(w))

	def _made_params(self):
		try:
			getattr(self.module, self.name + "_u")
			getattr(self.module, self.name + "_v")
			getattr(self.module, self.name + "_bar")
			return True
		except AttributeError:
			return False

	def _make_params(self):
		w = getattr(self.module, self.name)

		height = w.data.shape[0]
		width = w.view(height, -1).data.shape[1]

		u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
		v = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
		u.data = l2normalize(u.data)
		v.data = l2normalize(v.data)
		w_bar = Parameter(w.data)

		del self.module._parameters[self.name]
		self.module.register_parameter(self.name + "_u", u)
		self.module.register_parameter(self.name + "_v", v)
		self.module.register_parameter(self.name + "_bar", w_bar)

	def forward(self, *args):
		self._update_u_v()
		return self.module.forward(*args)
    
def l2normalize(v, eps=1e-4):
	return v / (v.norm() + eps)

class ConvSNLRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding=0, lrelu_slope=0.1):
        super().__init__()
        self.conv = SpectralNorm(nn.Conv2d(in_ch, out_ch, kernel, stride, padding=padding))
        self.lrelu = nn.LeakyReLU(lrelu_slope, True)
        
        self.conv.apply(init_xavier_uniform)

    def forward(self, inputs):
        return self.lrelu(self.conv(inputs))
    
    
def init_xavier_uniform(layer):
    if hasattr(layer, "weight"):
#        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.kaiming_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if hasattr(layer.bias, "data"):       
            layer.bias.data.fill_(0)

def init_xavier_uniform_real(layer):
    if hasattr(layer, "weight"):
        torch.nn.init.xavier_uniform_(layer.weight)
#        torch.nn.init.kaiming_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if hasattr(layer.bias, "data"):       
            layer.bias.data.fill_(0)


class DeconvBNRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding=0, n_classes=10, lrelu_slope=0.1):
        super().__init__()
#        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding=padding)
        self.conv = SpectralNorm(nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding=padding))
        self.bn = ConditionalBatchNorm2d(out_ch, n_classes)
        self.lrelu = nn.LeakyReLU(lrelu_slope, True)

        self.conv.apply(init_xavier_uniform)

    def forward(self, inputs, label_onehots):
        x = self.conv(inputs)
        x = self.bn(x, label_onehots)
        return self.lrelu(x)
    
class ConditionalBatchNorm2d(nn.Module):
  def __init__(self, num_features, num_classes, eps=1e-4, momentum=0.1):
    super().__init__()
    self.num_features = num_features
    self.bn = nn.BatchNorm2d(num_features, affine=False, eps=eps, momentum=momentum)
    self.gamma_embed = nn.Linear(num_classes, num_features, bias=False)
    self.beta_embed = nn.Linear(num_classes, num_features, bias=False)
    self.gamma_embed.apply(init_xavier_uniform)
    self.beta_embed.apply(init_xavier_uniform)
    

  def forward(self, x, y):
    out = self.bn(x)
#    print('y', y.size())
    gamma = self.gamma_embed(y) + 1
    beta = self.beta_embed(y)
    out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    return out


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax  = nn.Softmax(dim=-1) #

        self.query_conv.apply(init_xavier_uniform_real)
        self.key_conv.apply(init_xavier_uniform_real)
        self.value_conv.apply(init_xavier_uniform_real)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

