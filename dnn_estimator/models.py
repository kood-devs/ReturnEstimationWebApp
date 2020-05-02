from django.db import models

class LearningModel(models.Model):
    # identifier for past model
    title = models.CharField(max_length=100)

    # dnn params
    train_start = models.DateField()
    train_end = models.DateField()
    test_start = models.DateField()
    test_end = models.DateField()
    epoch = models.IntegerField()
    batch_size = models.IntegerField()

    def __str__(self):
        return self.title
    

