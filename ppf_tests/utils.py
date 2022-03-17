import os

def get_ansor_log_file(model_name, parameters, pass_context, target):
    batched_execution = pass_context.config["relay.db_batched_execution"]
    dynamic_batch_size_estimate = pass_context.config["relay.db_dynamic_batch_size_estimate"]
    config_str = ("%d_%d_%s") % (batched_execution, dynamic_batch_size_estimate, target)
    model_str = model_name + "_" + "_".join([str(i) for i in parameters])
    file_name = model_str + "_" + config_str + ".log"
    log_dir = "ansor_logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir + file_name
