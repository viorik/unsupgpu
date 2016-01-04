require 'torch'
require 'image'
--gfx = require 'graphicsmagick'

torch.setdefaulttensortype('torch.FloatTensor')

local function main()
  local std = std or 0.2
  local nsamples = 4967
  local nchannels = 1
  local nrows = 120
  local ncols = 160
  local gs = 5
  local trdata1 = torch.Tensor(nsamples,1,nrows-(gs-1),ncols-(gs-1)):zero()
  --local trdata1 = torch.Tensor(nsamples,1,nrows,ncols):zero()
  
  local function lcn(im)
    local gfh = image.gaussian{width=gs,height=1,normalize=true}
    local gfv = image.gaussian{width=1,height=gs,normalize=true}
    local gf = image.gaussian{width=gs,height=gs,normalize=true}

    local imsq = torch.Tensor()
    local lmnh = torch.Tensor()
    local lmn = torch.Tensor()
    local lmnsqh = torch.Tensor()
    local lmnsq = torch.Tensor()
    local lvar = torch.Tensor()
    local mn = im:mean()
    local std = im:std()

    im:add(-mn)
    im:div(std)
    imsq:resizeAs(im):copy(im):cmul(im)
    torch.conv2(lmnh,im,gfh)
    torch.conv2(lmn,lmnh,gfv)

    torch.conv2(lmnsqh,imsq,gfh)
    torch.conv2(lmnsq,lmnsqh,gfv)
    lvar:resizeAs(lmn):copy(lmn):cmul(lmn)
    lvar:mul(-1)
    lvar:add(lmnsq)
    lvar:apply(function (x) if x<0 then return 0 else return x end end)

    local lstd = lvar
    lstd:sqrt()
    lstd:apply(function (x) if x<1 then return 1 else return x end end)

    local shift = (gs+1)/2
    local nim = im:narrow(1,shift,im:size(1)-(gs-1)):narrow(2,shift,im:size(2)-(gs-1))
    nim:add(-1,lmn)
    nim:cdiv(lstd)
    return nim
  end
  
  
  for i=1,nsamples do
    print(i)
    local imName = string.format('MyDirectory/img_%04d.png',i) 
    local I = image.load(imName)
    -- local Ig = image.rgb2y(I:squeeze()) -- uncomment this if rgb images
    local Ig = I:float():squeeze()
    -- Ig = image.scale(Ig,nrows,ncols) -- uncomment this if not all images have same size
    local imn = lcn(Ig)
    imn = imn:reshape(1,imn:size(1),imn:size(2))
    trdata1[{{i},{},{},{}}]:copy(imn)
  end

  print(trdata1:size())

  torch.save('dataset_ascii.t7', trdata1, 'ascii')
  print ('done')
end
main()
