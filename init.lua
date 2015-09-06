require "nn"
require "optim"
require "libunsupgpu"

-- extra modules
include('unsupgpu', 'Diag.lua')
-- classes that implement algorithms
include('unsupgpu', 'UnsupModule.lua')
include('unsupgpu', 'FistaL1.lua')
include('unsupgpu', 'SpatialConvFistaL1.lua')
include('unsupgpu', 'psd.lua')
include('unsupgpu', 'ConvPsd.lua')
include('unsupgpu', 'UnsupTrainer.lua')

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
