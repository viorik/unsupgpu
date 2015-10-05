local SpatialConvFistaL1, parent = torch.class('unsupgpu.SpatialConvFistaL1','unsupgpu.FistaL1')
-- conntable       : Connection table (ref: nn.SpatialConvolutionMap)
-- kw              : width of convolutional kernel
-- kh              : height of convolutional kernel
-- iw              : width of input patches
-- ih              : height of input patches
-- padw            : zero padding horizontal
-- padh            : zero padding vertical
-- lambda          : sparsity coefficient
-- params          : optim.FistaLS parameters
function SpatialConvFistaL1:__init(conntable, kw, kh, iw, ih, padw, padh, lambda, batchSize)

   -- parent.__init(self)

   -----------------------------------------
   -- dictionary is a linear layer so that I can train it
   -----------------------------------------
   --local D = nn.SpatialFullConvolutionMap(conntable, kw, kh, 1, 1)
   local outputFeatures = conntable:select(2,1):max()
   local inputFeatures = conntable:select(2,2):max()
   local D = nn.SpatialConvolution(outputFeatures, inputFeatures, kw, kh, 1, 1, padw, padh)
   -----------------------------------------
   -- L2 reconstruction cost with weighting
   -----------------------------------------
   local batchSize = batchSize or 1
   local tt, utt
   if batchSize > 1 then 
     tt = torch.Tensor(batchSize,inputFeatures,ih,iw)
     utt= tt:unfold(3,kh,1):unfold(4,kw,1)
   else
     tt = torch.Tensor(inputFeatures,ih,iw)
     utt= tt:unfold(2,kh,1):unfold(3,kw,1)
   end
   tt:zero()
   utt:add(1)
   tt:div(tt:max())
   local Fcost = nn.WeightedMSECriterion(tt)
   Fcost.sizeAverage = false;

   parent.__init(self,D,Fcost,lambda)

   -- this is going to be passed to optim.FistaLS
   if batchSize > 1 then
     self.code:resize(batchSize, outputFeatures, utt:size(3)+2*padw,utt:size(4)+2*padh)
   else
     self.code:resize(outputFeatures, utt:size(2)+2*padw,utt:size(3)+2*padh)
   end
   self.code:fill(0)
end

