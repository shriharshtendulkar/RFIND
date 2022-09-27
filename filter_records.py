#!/opt/miniconda3/bin/python
import numpy as np 
import click 

@click.command()
@click.argument('inputfile', type=click.File('rb'))
@click.argument('outputfile', type=click.File('wb'))
@click.option("--min_snr", default=10, type=click.FLOAT, help='Minimum SNR threshold for the delay fringes')
@click.option("--max_rec", default=500, type=click.INT, help="Choose the brightest max_rec fringes")
@click.option("--drop_ants", "-d", multiple=True, type=click.INT)
def filter_records(inputfile, outputfile, 
    min_snr, max_rec, drop_ants):
    """
    Commandline filtering for records
    """

    print("inputs: {} {} {} {}".format(inputfile, outputfile, 
    min_snr, max_rec, drop_ants))

    data = np.load(inputfile)

    delays = data['delays']

    filt = np.where(delays['amplitude']>min_snr)

    delays = delays[filt]

    delays.sort(order='amplitude')

    print("Num_records: {}".format(len(delays)))

    np.savez(outputfile,
    positions=data['positions'],
    baselines=data['baselines'],
    delays=delays[-max_rec:-1]
    )



if __name__=="__main__":
    filter_records()