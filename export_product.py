'''
export necessary components for production
'''
import os
import click


@click.command()
@click.option('-d','--dest',help='Target Destination',type=str)
def export(dest):

    try:
        dest_d = dest+'ASR/'

        if not os.path.isdir(dest):
            os.makedirs(dest)
        if not os.path.isdir(dest_d):
            os.makedirs(dest_d)

        file=['ASR/asr.py','ASR/lm.py'] # replaceable python file
        for f in file:
            os.system(f'cp {f} {dest+f}')

        file_no_replace=['config.py'] # non-replace python file
        for f in file_no_replace:
            if not os.path.isfile((dest+f)):
                os.system(f'cp {f} {dest+f}')
            else:
                print(f'[info] {f} already exist in target destination, no replacement')

        os.system(f'cp -R ASR/file_function {dest_d}') #file_function
        os.system(f'cp -R ASR/dict {dest_d}') #dict
        os.system(f'cp -R ASR/bg {dest_d}') #bg

        print(f'[info] Exported to: {dest}')

    except Exception as e:
        print(e)

if __name__=="__main__":
    export()