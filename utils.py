import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

def ema(target_model, model, decay=0.99):
    shadow_target = dict()
    for name, param in target_model.named_parameters():
            if param.requires_grad:
                shadow[name] = param.data.clone()
                shadow[name] = (1-decay)