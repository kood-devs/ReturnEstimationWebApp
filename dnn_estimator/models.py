from django.db import models
from django.utils import timezone


class LearningModel(models.Model):
    # identifier for past model
    title = models.CharField(max_length=100)
    model_dev_date = models.DateField(default=timezone.now)

    # dnn input params
    train_start = models.DateField()
    train_end = models.DateField()
    test_start = models.DateField()
    test_end = models.DateField()
    epoch = models.IntegerField()
    batch_size = models.IntegerField()

    # dnn results
    train_acc = models.FloatField(default=0.0)
    test_acc = models.FloatField(default=0.0)

    def __str__(self):
        return self.title
