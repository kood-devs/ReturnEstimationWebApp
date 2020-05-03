# Generated by Django 3.0.5 on 2020-05-03 11:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dnn_estimator', '0002_learningmodel_title'),
    ]

    operations = [
        migrations.AddField(
            model_name='learningmodel',
            name='test_acc',
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='learningmodel',
            name='train_acc',
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
    ]