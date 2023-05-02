
def make_optim_paramgroup(model, lr, decoder_lr_times):
    if decoder_lr_times:
        param_list = []
        all_modules = model.get_all_modules()
        param_list.append(dict(params=all_modules.pop("encoder").parameters(), lr=lr))
        all_modules = list(all_modules.values())
        [param_list.append(dict(params=m.parameters(), lr=lr*decoder_lr_times)) for m in all_modules]
        return param_list
        