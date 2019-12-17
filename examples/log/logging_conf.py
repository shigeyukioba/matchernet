def set_logger_config(file_path):
    from json import load
    from logging import config, Filter

    class VisualizeFilter(Filter):

        def __init__(self, words=None):
            if not isinstance(words, list):
                words = None
            self.words = words

        def word_filter(self, record):
            if self.words is not None:
                for word in self.words:
                    if word in record.msg:
                        record.msg = "Filterd: " + word
                        break
            return True

    with open(file_path, "r", encoding="utf-8") as f:
        log_conf_dic = load(f)

    log_conf_dic["filters"]["visualizeFilter"]["()"] = VisualizeFilter

    config.dictConfig(log_conf_dic)
