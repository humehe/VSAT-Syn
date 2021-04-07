from itertools import tee, islice, chain

from Fnc_Syn_Dir import *

####Fnc_Stk_Utl###
def Delete_Element_Array(array2bused,string2bdeleted):
	element2bdeleted = np.where(array2bused==string2bdeleted)
	array2bused = np.delete(array2bused, element2bdeleted)
	return array2bused

def Prev_Next(some_iterable,*args, **kwargs):
    prevs, items, nexts = tee(some_iterable, 3)
    prevs = chain([None], prevs)
    nexts = chain(islice(nexts, 1, None), [None])
    return zip(prevs, items, nexts)

def Def_Sub_Dirs_Slice_xtr(Prt_Dir,Slices,*args, **kwargs):
	sub_dir_slc  = []
	slices_split = []
	sub_dir_slc.append(Prt_Dir + str(Slices[0]) + '-' + str(Slices[-1]) + '/')
	slices_split.append(str(Slices[0]) + '-' + str(Slices[-1]))
	return sub_dir_slc,slices_split

def Def_Sub_Dirs_Slice_all(Prt_Dir,Slices,*args, **kwargs):
	sub_dir_slc  = []
	slices_split = []
	for previous, item, nxt in Prev_Next(Slices):
		slc_int_bin = Slices[1]-Slices[0]
		if item < Slices[-1]:
		   sub_dir_slc.append(Prt_Dir + str(item) + '-' + str(nxt) + '/')
		   slices_split.append(str(item) + '-' + str(nxt))
		else:
		    break
	sub_dir_slc.append(Prt_Dir + str(Slices[0]) + '-' + str(Slices[-1]) + '/')
	slices_split.append(str(Slices[0]) + '-' + str(Slices[-1]))
	return sub_dir_slc,slices_split
####Fnc_Stk_Utl###