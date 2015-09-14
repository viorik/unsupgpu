require "nn"
require "optim"
require "libunsupgpu"

-- extra modules
torch.include('unsupgpu','Diag.lua')
-- classes that implement algorithms
torch.include('unsupgpu','UnsupgpuModule.lua')
torch.include('unsupgpu','FistaL1.lua')
torch.include('unsupgpu','SpatialConvFistaL1.lua')
torch.include('unsupgpu','psd.lua')
torch.include('unsupgpu','ConvPsd.lua')
torch.include('unsupgpu','UnsupgpuTrainer.lua')

local oldhessian = nn.hessian.enable
function nn.hessian.enable()
	oldhessian() -- enable Hessian usage
	----------------------------------------------------------------------
	-- Diag
	----------------------------------------------------------------------
	local accDiagHessianParameters = nn.hessian.accDiagHessianParameters
	local updateDiagHessianInput = nn.hessian.updateDiagHessianInput
	local updateDiagHessianInputPointWise = nn.hessian.updateDiagHessianInputPointWise
	local initDiagHessianParameters = nn.hessian.initDiagHessianParameters

	function nn.Diag.updateDiagHessianInput(self, input, diagHessianOutput)
	   updateDiagHessianInput(self, input, diagHessianOutput, {'weight'}, {'weightSq'})
	   return self.diagHessianInput
	end

	function nn.Diag.accDiagHessianParameters(self, input, diagHessianOutput)
	   accDiagHessianParameters(self,input, diagHessianOutput, {'gradWeight'}, {'diagHessianWeight'})
	end

	function nn.Diag.initDiagHessianParameters(self)
	   initDiagHessianParameters(self,{'gradWeight'},{'diagHessianWeight'})
	end
end

