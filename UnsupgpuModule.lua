local UnsupgpuModule,parent = torch.class('unsupgpu.UnsupgpuModule','nn.Module')

function UnsupgpuModule:__init()
	parent.__init(self)
end

function UnsupgpuModule:normalize()
	error('Every unsupervised module has to implement normalize function')
end

