using MLJ
using StatisticalMeasures

function auc_one_vs_all(yhat_prob, gt, class_name)
    scores = pdf.(yhat_prob, class_name)
    return StatisticalMeasures.Functions.auc(scores, gt, class_name)
end
function multiclass_auc(prob_preds::Vector{<:UF}, gt::Vector{Int}) where {UF <: UnivariateFinite}
    @assert length(prob_preds) == length(gt)
    classes = levels(first(prob_preds))
    aucs = [auc_one_vs_all(prob_preds, gt, c) for c in classes]
    macro_auc = mean(aucs)
    return macro_auc, aucs
end
