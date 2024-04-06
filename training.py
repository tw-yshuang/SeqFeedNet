import click

help_doc = {
    'se_network': "Sequence Extract Network",
    'me_network': "Mask Extract Network",
    'sm_network': "Sequence to mask Network",
    'loss_func': "Please check utils/evaluate/losses.py to find others",
    'optimizer': "Optimizer that provide by Pytorch",
    'learning_rate': "Learning Rate for optimizer",
    'weight_decay': "Weight Decay for optimizer",
    'num_epochs': "Number of epochs",
    'batch_size': "Number of batch_size",
    'num_workers': "Number of workers for data processing",
    'cv_set_number': "Cross validation set number for training and test videos will be selected",
    'img_sizeHW': "Image size for training",
    'data_split_rate': "Split data to train_set & val_set",
    'use_test_as_val': "Use test_data as validation data, use this flag will set '--data_split_rate=1.0'",
    'device': "CUDA ID, if system can not find Nvidia GPU, it will use CPU",
    'do_testing': "Do testing evaluation is a time-consuming process, suggest not do it",
    'pretrain_weight': "Pretrain weight, model structure must same with the setting",
    'output': "Model output directory",
}


@click.command(context_settings=dict(help_option_names=['-h', '--help'], max_content_width=120))
@click.option('-se', '--se_network', default='UNetVgg16', help=help_doc['se_network'])
@click.option('-me', '--me_network', default='UNetVgg16', help=help_doc['me_network'])
@click.option('-sm', '--sm_network', default='SMNet2D', help=help_doc['sm_network'])
@click.option('-loss', '--loss_func', default='IOULoss4CDNet2014', help=help_doc['loss_func'])
@click.option('-opt', '--optimizer', default='', help=help_doc['optimizer'])
@click.option('-lr', '--learning_rate', default=1e-4, help=help_doc['learning_rate'])
@click.option('-wd', '--weight_decay', default=0.0, help=help_doc['weight_decay'])
@click.option('-epochs', '--num_epochs', default=0, help=help_doc['num_epochs'])
@click.option('-bs', '--batch_size', default=8, help=help_doc['batch_size'])
@click.option('-workers', '--num_workers', default=1, help=help_doc['num_workers'])
@click.option('-cv', '--cv_set_number', default=1, help=help_doc['cv_set_number'])
@click.option('-imghw', '--img_sizeHW', 'img_sizeHW', default='224-224', help=help_doc['img_sizeHW'])
@click.option('-drate', '--data_split_rate', default=1.0, help=help_doc['data_split_rate'])
@click.option('-use-t2val', '--use_test_as_val', default=False, is_flag=True, help=help_doc['use_test_as_val'])
@click.option('--device', default=0, help=help_doc['device'])
@click.option('--do_testing', default=False, is_flag=True, help=help_doc['do_testing'])
@click.option('--pretrain_weight', default='', help=help_doc['pretrain_weight'])
@click.option('-out', '--output', default='', help=help_doc['output'])
def cli(
    se_network: str,
    me_network: str,
    sm_network: str,
    loss_func: str,
    optimizer: str,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    batch_size: int,
    num_workers: int,
    cv_set_number: int,
    img_sizeHW: str,
    data_split_rate: float,
    use_test_as_val: bool,
    device: int,
    do_testing: bool,
    pretrain_weight: str,
    output: str,
):
    from Processor import convertStr2Parser, execute

    parser = convertStr2Parser(
        se_network,
        me_network,
        sm_network,
        loss_func,
        optimizer,
        learning_rate,
        weight_decay,
        num_epochs,
        batch_size,
        num_workers,
        cv_set_number,
        img_sizeHW,
        data_split_rate,
        use_test_as_val,
        device,
        do_testing,
        pretrain_weight=pretrain_weight,
        output=output,
    )

    execute(parser)


if __name__ == '__main__':
    # import sys

    # sys.argv = 'training.py --device 1 -epochs 0 -workers 2 -cv 2 --pretrain_weight out/1211-0444_iouLoss.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-27_Set-2/bestAcc-F_score.pt --do_testing'.split()
    # sys.argv = 'training.py --device 1 -epochs 2 --batch_size 8 -workers 8 -cv 2 -imghw 112-112 -use-t2val -opt Adam -wd 0.01 -out test'.split()
    # sys.argv = 'training.py --device 7 -epochs 119 --batch_size 9 -workers 9 -cv 4 -imghw 224-224 -use-t2val --pretrain_weight out/0130-1336_dev4.maxGAP15.cv4.224_SMNet2D.UNetVgg16-UNetVgg16_AdamW1.0e-04.wd0.0_IOULoss_BS-9_Set-4/checkpoint_e080.pt -out dev4.maxGAP15.cv2.to200.224'.split()
    cli()
