from django.db import models
from django.utils import timezone


class LearningModel(models.Model):
    # identifier for past model
    title = models.CharField('モデル名', max_length=100)
    model_dev_date = models.DateField(default=timezone.now())

    # dnn input params
    train_start = models.DateField('訓練データ開始日')
    train_end = models.DateField('訓練データ終了日')
    test_start = models.DateField('テストデータ開始日')
    test_end = models.DateField('テストデータ終了日')
    epoch = models.IntegerField('エポック数')
    batch_size = models.IntegerField('バッチサイズ')

    # dnn results
    train_acc = models.FloatField('訓練データ正答率', default=0.0)
    test_acc = models.FloatField('テストデータ正答率', default=0.0)

    def __str__(self):
        return self.title
