import click

help_doc = {
    'pretrain_weight': "Pretrain weight, model structure must same with the setting",
    'cv_set_number': "Cross validation set number for training and test videos will be selected",
    'only_temporalROI': "Only do testing evaluation during temporalROI",
    'save_result': "Save all the predict results",
    'device': "CUDA ID, if system can not find Nvidia GPU, it will use CPU",
    'output': "Model output directory",
}


@click.command(context_settings=dict(help_option_names=['-h', '--help'], max_content_width=120))
@click.option('-weight', '--pretrain_weight', default='', help=help_doc['pretrain_weight'])
@click.option('-cv', '--cv_set_number', default=1, help=help_doc['cv_set_number'])
@click.option('-onlyROI', '--only_temporal_roi', default=False, is_flag=True, help=help_doc['only_temporalROI'])
@click.option('-save', '--save_result', default=False, is_flag=True, help=help_doc['save_result'])
@click.option('--device', default=0, help=help_doc['device'])
@click.option('-out', '--output', default='', help=help_doc['output'])
def cli(
    cv_set_number: int,
    device: int,
    only_temporal_roi: bool,
    save_result: bool,
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
        test_from_begin=~only_temporal_roi,
        save_test_result=save_result,
        output=output,
    )

    execute(parser)


if __name__ == '__main__':
    # import sys

    # sys.argv = ' testing.py -cv 4 --device 4 -weight out/0119-0949_dev3.adamW.maxGAP15.cv4.224_SMNet2D.UNetVgg16-UNetVgg16_AdamW1.0e-04.wd0.0_IOULoss_BS-9_Set-4/bestAcc-F_score.pt -save'.split()
    cli()
