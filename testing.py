import click

help_doc = {
    'pretrain_weight': "Pretrain weight, model structure must same with the setting",
    'cv_set_number': "Cross validation set number for training and test videos will be selected",
    'device': "CUDA ID, if system can not find Nvidia GPU, it will use CPU",
    'output': "Model output directory",
}


@click.command(context_settings=dict(help_option_names=['-h', '--help'], max_content_width=120))
@click.option('-weight', '--pretrain_weight', default='', help=help_doc['pretrain_weight'])
@click.option('-cv', '--cv_set_number', default=1, help=help_doc['cv_set_number'])
@click.option('--device', default=0, help=help_doc['device'])
@click.option('-out', '--output', default='', help=help_doc['output'])
def cli(
    cv_set_number: int,
    device: int,
    pretrain_weight: str,
    output: str,
):
    from Processor import convertStr2Parser, execute

    parser = convertStr2Parser(
        num_epochs=0,
        do_testing=True,
        cv_set_number=cv_set_number,
        device=device,
        pretrain_weight=pretrain_weight,
        output=output,
    )

    execute(parser)


if __name__ == '__main__':
    # import sys

    # sys.argv = 'testing.py --device 0 -cv 5 --pretrain_weight out/1211-0444_iouLoss.112_SMNet2D.UNetVgg16-UNetVgg16_Adam1.0e-04_IOULoss_BS-27_Set-2/checkpoint_e170.pt -out test'.split()
    cli()
