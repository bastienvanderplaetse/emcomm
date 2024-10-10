def EMA_score(model, decay):
    model.ema_score = (decay * model.ema_score) + (1-decay) * model.avg_score

class EMA():
    def __init__(self, model, decay):
        self.model = model 
        self.decay = decay
        self.shadow = dict()
        self._register()
    
    def _register(self):
        self.register(self.model, self.shadow)

    def register(self, model, shadow):
        for name, param in model.named_parameters():
            if param.requires_grad:
                shadow[name] = param.data.clone()


    def update(self, model):
        model_shadow = dict()
        self.register(model, model_shadow)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1 - self.decay) * model_shadow[name] + self.decay * self.shadow[name]
                param.data = self.shadow[name]
