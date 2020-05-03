from django.db import models

class LearningModel(models.Model):
    # identifier for past model
    title = models.CharField(max_length=100)

    # dnn input params
    train_start = models.DateField()
    train_end = models.DateField()
    test_start = models.DateField()
    test_end = models.DateField()
    epoch = models.IntegerField()
    batch_size = models.IntegerField()

    # dnn results
    train_acc = models.FloatField()
    test_acc = models.FloatField()

    def __str__(self):
        return self.title
    

