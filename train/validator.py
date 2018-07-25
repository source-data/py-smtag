# -*- coding: utf-8 -*-
#T. Lemberger, 2018

class validator:
    def __init__():
    
    
    def run(model, minibatches):
        
    local criterion = CONFIG.CRITERION
    local binarize = binarize or false
    local token_level = token_level or false
    local nf = minibatches[1].output:size(2)
    net:evaluate()
    local p_sum = torch.Tensor(nf):fill(1e-12)
    local tp_sum = torch.Tensor(nf):fill(1e-12)
    local fp_sum = torch.Tensor(nf):fill(1e-12)
    local p_sum_sum = 1e-12
    local tp_sum_sum = 1e-12
    local fp_sum_sum = 1e-12
    if CUDA_ON then
        p_sum = p_sum:cuda()
        tp_sum = tp_sum:cuda()
        fp_sum = fp_sum:cuda()
    end
    local losses = {}
    local default_threshold = CONFIG.THRESHOLDS['default'].character
    local attrmap = attrmap or {}
    -- initialize attrmap to default for each feature if no specific attrmap provided
    if #attrmap == 0 then
        for i = 1, nf do attrmap[i] = {'default'} end
    end
    
    for i = 1, #minibatches do
		--print("minibatch:", i)
		local input = minibatches[i].input
		local target = minibatches[i].output
		
		local prediction = net:forward(input)
		local loss = criterion:forward(prediction, target)
		
		local p, tp, fp, precision, recall, f1 --1D tensors with size(1) = nf
		if binarize then
		    local binarized_prediction, binarized_target
		    if token_level then 
		        local tokenized = minibatches[i].tokenized
		        binarized_prediction = smtag.binarize_token(tokenized, prediction, attrmap)
		        binarized_target = smtag.binarize_token(tokenized, target, attrmap)
		        p, tp, fp = validator.tpfp(binarized_prediction.start, binarized_target.start, 0.9)
		    else
				binarized_prediction = smtag.binarize(prediction, attrmap)
				p, tp, fp = validator.tpfp(binarized_prediction.marks, target, 0.9)
		    end
		else
		    p, tp, fp = validator.tpfp(prediction, target, default_threshold)
		end
		p_sum = torch.add(p_sum, p); p_sum_sum = p_sum_sum + p_sum:sum()
		tp_sum = torch.add(tp_sum, tp); tp_sum_sum = tp_sum_sum + tp_sum:sum()
		fp_sum = torch.add(fp_sum, fp); fp_sum_sum = fp_sum_sum + fp_sum:sum() 
		table.insert(losses, loss)
    end
    net:training()
    local p_avg = torch.zeros(nf)
    local r_avg = torch.zeros(nf) 
    local f1_avg = torch.zeros(nf)
    local l_avg = torch.zeros(nf) 
    if #losses > 0 then l_avg = torch.Tensor(losses):mean() end
    p_avg = torch.cdiv(tp_sum, torch.add(tp_sum, fp_sum))
    r_avg = torch.cdiv(tp_sum, p_sum)
    f1_avg = torch.cdiv(torch.mul(torch.cmul(r_avg , p_avg),2), torch.add(r_avg, p_avg))
    local p_avg_global = tp_sum_sum / (tp_sum_sum + fp_sum_sum)
    local r_avg_global = tp_sum_sum / p_sum_sum 
    local f1_avg_global = 2 * p_avg_global * r_avg_global / (p_avg_global + r_avg_global)
    
    --debug.sethook()
    
    return p_avg, r_avg, f1_avg, l_avg, p_avg_global, r_avg_global, f1_avg_global

end