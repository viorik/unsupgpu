local ConvPSD, parent = torch.class('unsupgpu.ConvPSD','unsupgpu.PSD')

-- conntable       : A connection table (ref nn.SpatialConvolutionMap)
-- kw, kh          : width, height of convolutional kernel
-- iw, ih          : width, height of input patches
-- lambda          : sparsity coefficient
-- beta            : prediction coefficient
-- params          : optim.FistaLS parameters
function ConvPSD:__init(conntable, kw, kh, iw, ih, lambda, beta, params)
   
   -- prediction weight
   self.beta = beta

   local decodertable = conntable:clone()
   decodertable:select(2,1):copy(conntable:select(2,2))
   decodertable:select(2,2):copy(conntable:select(2,1))
   local outputFeatures = conntable:select(2,2):max()
   local padw = torch.floor(kw/2.0)
   local padh = torch.floor(kh/2.0)

   -- decoder is L1 solution
   self.decoder = unsup.SpatialConvFistaL1(decodertable, kw, kh, iw, ih, padw, padh, lambda, params)

   -- encoder
   params = params or {}
   self.params = params

   self.encoder = nn.Sequential()
   self.encoder:add(nn.SpatialConvolution(conntable, kw, kh, 1, 1, padw, padh))
   self.encoder:add(nn.Tanh())
   self.encoder:add(nn.Diag(outputFeatures))

   parent.__init(self, self.encoder, self.decoder, beta, params)
end

