require 'image'
require 'optim'
require 'cunn'
require 'unsupgpu'
require 'autoencoder-data'
require 'torch'
require 'cutorch'

----------------------------------------------------------------------
-- parse command-line options
--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple sparse coding dictionary on Berkeley images')
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-dir', 'outputs', 'subdirectory to save experiments in')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 1, 'threads')

-- for all models:
cmd:option('-model', 'conv-psd', 'auto-encoder class: linear | linear-psd | conv | conv-psd')
cmd:option('-inputsize', 25, 'size of each input patch')
cmd:option('-nfiltersin', 1, 'number of input convolutional filters')
cmd:option('-nfiltersout', 16, 'number of output convolutional filters')
cmd:option('-lambda', 1, 'sparsity coefficient')
cmd:option('-beta', 1, 'prediction error coefficient')
cmd:option('-eta', 2e-3, 'learning rate')
cmd:option('-batchsize', 5, 'batch size')
cmd:option('-etadecay', 1e-5, 'learning rate decay')
cmd:option('-momentum', 0.9, 'gradient momentum')
cmd:option('-maxiter', 1000000, 'max number of updates')

-- use hessian information for training:
cmd:option('-hessian', true, 'compute diagonal hessian coefficients to condition learning rates')
cmd:option('-hessiansamples', 500, 'number of samples to use to estimate hessian')
cmd:option('-hessianinterval', 10000, 'compute diagonal hessian coefs at every this many samples')
cmd:option('-minhessian', 0.02, 'min hessian to avoid extreme speed up')
cmd:option('-maxhessian', 500, 'max hessian to avoid extreme slow down')

-- for conv models:
cmd:option('-kernelsize', 9, 'size of convolutional kernels')

-- logging:
cmd:option('-datafile','http://torch7.s3-website-us-east-1.amazonaws.com/data/tr-berkeley-N5K-M56x56-lcn.ascii', 'Dataset URL')
cmd:option('-statinterval', 5000, 'interval for saving stats and models')
cmd:option('-v', false, 'be verbose')
cmd:option('-display', true, 'display stuff')
cmd:option('-wcar', '', 'additional flag to differentiate this run')
cmd:text()

params = cmd:parse(arg)

rundir = cmd:string('psd', params, {dir=true})
params.rundir = params.dir .. '/' .. rundir

if paths.dirp(params.rundir) then
   os.execute('rm -r ' .. params.rundir)
end
os.execute('mkdir -p ' .. params.rundir)
cmd:addTime('psd')
cmd:log(params.rundir .. '/log.txt', params)

torch.setdefaulttensortype('torch.FloatTensor')

cutorch.setDevice(2) -- by default, use GPU 1
torch.manualSeed(params.seed)
local statinterval = torch.floor(params.statinterval / params.batchsize)*params.batchsize
local hessianinterval = torch.floor(params.hessianinterval / params.batchsize)*params.batchsize
print (statinterval)
print (hessianinterval)


--torch.manualSeed(params.seed)

torch.setnumthreads(params.threads)

----------------------------------------------------------------------
-- load data
--
filename = paths.basename(params.datafile)
print (filename)
if not paths.filep(filename) then
   os.execute('wget ' .. params.datafile .. '; '.. 'tar xvf ' .. filename)
end
dataset = getdata(filename, params.inputsize)


if params.display then
   displayData(dataset, 100, 10, 2)
end

----------------------------------------------------------------------
-- create model
--
-- params:
   local conntable = nn.tables.full(params.nfiltersin, params.nfiltersout)
   local kw, kh = params.kernelsize, params.kernelsize
   local H, W = params.inputsize, params.inputsize
   local padw, padh = torch.floor(params.kernelsize/2.0), torch.floor(params.kernelsize/2.0)
   local batchSize = params.batchsize or 1
   -- connection table:
   local decodertable = conntable:clone()
   decodertable[{ {},1 }] = conntable[{ {},2 }]
   decodertable[{ {},2 }] = conntable[{ {},1 }] 
   local outputFeatures = conntable[{ {},2 }]:max()
   local inputFeatures = conntable[{ {},1 }]:max()

   -- encoder:
   encoder = nn.Sequential()
   encoder:add(nn.SpatialConvolution(inputFeatures,outputFeatures, kw, kh, 1, 1, padw, padh))
   encoder:add(nn.Tanh())
   
   encoder:add(nn.Diag(outputFeatures))

   -- decoder is L1 solution:
   decoder = unsupgpu.SpatialConvFistaL1(decodertable, kw, kh, W, H, padw, padh, params.lambda, batchSize)

   -- PSD autoencoder
   module = unsupgpu.PSD(encoder, decoder, params.beta)

   module:cuda()
   -- convert dataset to convolutional (returns 1xKxK tensors (3D), instead of K*K (1D))
   dataset:conv()

   -- verbose
   print('==> constructed convolutional predictive sparse decomposition (PSD) auto-encoder')

----------------------------------------------------------------------
-- trainable parameters
--

-- are we using the hessian?
if params.hessian then
   nn.hessian.enable()
   module:initDiagHessianParameters()
end

-- get all parameters
x,dl_dx,ddl_ddx = module:getParameters()

----------------------------------------------------------------------
-- train model
--

print('==> training model')

local avTrainingError = torch.FloatTensor(math.ceil(params.maxiter/params.statinterval)):zero()
local err = 0
local iter = 0

for t = 1,params.maxiter,params.batchsize do

   --------------------------------------------------------------------
   -- update diagonal hessian parameters
   --
   if params.hessian and math.fmod(t , hessianinterval) == 1 then
      -- some extra vars:
      local batchsize = params.batchsize
      local hessiansamples = params.hessiansamples
      local minhessian = params.minhessian
      local maxhessian = params.maxhessian
      local ddl_ddx_avg = ddl_ddx:clone(ddl_ddx):zero()
      etas = etas or ddl_ddx:clone()

      print('==> estimating diagonal hessian elements')
      
      for ih = 1,hessiansamples,batchsize do
        local inputs = torch.Tensor(params.batchsize,params.nfiltersin,params.inputsize,params.inputsize)
        local targets = torch.Tensor(params.batchsize,params.nfiltersin,params.inputsize,params.inputsize)
        for i = ih,ih+batchsize-1 do
          -- next
          local ex = dataset[i]
          local input = ex[1]
          local target = ex[2]
          inputs[{i-ih+1,{},{},{}}] = input
          targets[{i-ih+1,{},{},{}}] = target--[{{},{5,21},{5,21}}]
        end
   
        local inputs_ = inputs:cuda()
        local targets_ = targets:cuda()
        module:updateOutput(inputs_, targets_)
        --print ('Done one fw pass')
        -- gradient
        dl_dx:zero()
        module:updateGradInput(inputs_, targets_)
        module:accGradParameters(inputs_, targets_) 

        -- hessian
        ddl_ddx:zero()
        module:updateDiagHessianInput(inputs_, targets_)
        module:accDiagHessianParameters(inputs_, targets_)

        -- accumulate
        ddl_ddx_avg:add(batchsize/hessiansamples, ddl_ddx)
      end

      -- cap hessian params
      print('==> ddl/ddx : min/max = ' .. ddl_ddx_avg:min() .. '/' .. ddl_ddx_avg:max())
      ddl_ddx_avg[torch.lt(ddl_ddx_avg,minhessian)] = minhessian
      ddl_ddx_avg[torch.gt(ddl_ddx_avg,maxhessian)] = maxhessian
      print('==> corrected ddl/ddx : min/max = ' .. ddl_ddx_avg:min() .. '/' .. ddl_ddx_avg:max())

      -- generate learning rates
      etas:fill(1):cdiv(ddl_ddx_avg)
      collectgarbage()
   end

   --------------------------------------------------------------------
   -- progress
   --
   iter = iter+1
   xlua.progress(iter*params.batchsize, params.statinterval)

   --------------------------------------------------------------------
   -- create mini-batch
   --
   local example = dataset[t]
   local inputs = torch.Tensor(params.batchsize,params.nfiltersin,params.inputsize,params.inputsize)
   local targets = torch.Tensor(params.batchsize,params.nfiltersin,params.inputsize,params.inputsize)
   for i = t,t+params.batchsize-1 do
      -- load new sample
      local sample = dataset[i]
      local input = sample[1]:clone()
      local target = sample[2]:clone()
      inputs[{i-t+1,{},{},{}}] = input
      targets[{i-t+1,{},{},{}}] = target
   end
   
   inputs_ = inputs:cuda()
   targets_ = targets:cuda()
   --------------------------------------------------------------------
   -- define eval closure
   --
   local feval = function()
      -- reset gradient/f
      local f = 0
      dl_dx:zero()

      -- estimate f and gradients, for minibatch
      f = f + module:updateOutput(inputs_, targets_)

      -- gradients
      module:updateGradInput(inputs_, targets_)
      module:accGradParameters(inputs_, targets_)
      
      -- normalize
      --dl_dx:div(#inputs)
      --f = f/#inputs
      dl_dx:div(inputs:size(1))
      f = f/(inputs:size(1))

      -- return f and df/dx
      return f,dl_dx
   end

   --------------------------------------------------------------------
   -- one SGD step
   --
   sgdconf = sgdconf or {learningRate = params.eta,
                         learningRateDecay = params.etadecay,
                         learningRates = etas,
                         momentum = params.momentum}
   _,fs = optim.sgd(feval, x, sgdconf)
   err = err + fs[1]

   -- normalize
   if params.model:find('psd') then
      module:normalize()
   end

   --------------------------------------------------------------------
   -- compute statistics / report error
   --print (t .. ' ' .. statinterval .. ' ' .. math.fmod(t , statinterval) )
   if math.fmod(t , statinterval) == 1 then

      -- report
      print('==> iteration = ' .. t .. ', average loss = ' .. err/params.statinterval)

      -- get weights
      eweight = module.encoder.modules[1].weight
      if module.decoder.D then
         dweight = module.decoder.D.weight
      else
         dweight = module.decoder.modules[1].weight
      end

      -- reshape weights if linear matrix is used
      if params.model:find('linear') then
         dweight = dweight:transpose(1,2):unfold(2,params.inputsize,params.inputsize)
         eweight = eweight:unfold(2,params.inputsize,params.inputsize)
      end
      dweight_cpu = dweight:float():view(params.nfiltersout, params.kernelsize, params.kernelsize)
      eweight_cpu = eweight:float():view(params.nfiltersout, params.kernelsize, params.kernelsize)
      --print(dweight:type())
      --print(dweight:size())

      -- render filters
      dd = image.toDisplayTensor{input=dweight_cpu,
                                 padding=2,
                                 nrow=math.floor(math.sqrt(params.nfiltersout)),
                                 symmetric=true}
      de = image.toDisplayTensor{input=eweight_cpu,
                                 padding=2,
                                 nrow=math.floor(math.sqrt(params.nfiltersout)),
                                 symmetric=true}

      -- live display
      if params.display then
         _win1_ = image.display{image=dd, win=_win1_, legend='Decoder filters', zoom=2}
         _win2_ = image.display{image=de, win=_win2_, legend='Encoder filters', zoom=2}
      end

      -- save stuff
      image.save(params.rundir .. '/filters_dec_' .. t .. '.jpg', dd)
      image.save(params.rundir .. '/filters_enc_' .. t .. '.jpg', de)
      
      
      torch.save(params.rundir .. '/model_' .. t .. '.bin', module)
      torch.save(params.rundir .. '/sgdconf_' .. t .. '.bin', sgdconf)
      collectgarbage()
      -- reset counters
      err = 0; iter = 0
   end
end
