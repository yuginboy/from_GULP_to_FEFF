"""
note on np.unique

this module expands on a the functionality one would currently bootstrap from np.unique
to facilitate the functionality implemented in this module, a generalization of unique is proposed
that is, the version of unique implemented here also takes a lex-sortable tuple,
or multidimensional array viewed along one of its axes as a valid input; ie:

    #stack of 4 images
    images = np.random.rand(4,64,64)
    #shuffle the images; this is a giant mess now; how to find the unique ones?
    shuffled = images[np.random.randint(0,4,200)]
    #there you go
    print unique(shuffled, axis=0)

furthermore, some np.unique related functions are proposed, with a more friendly interface
(multiplicity, count)
"""


"""
A note on pandas
This module has substantial overlap with pandas' grouping functionality.
So whats the reason for implementing it in numpy?

Primarily; the concept of grouping is far more general than pandas' dataframe.
There is no reason why numpy ndarrays should not have a solid core of grouping functionality.

The recently added ufunc support make that we can now express grouping operations in pure numpy;
that is, without any slow python loops or cumbersome C-extensions.

I was not aware of pandas functionalty on this front when starting working on it;
so i am somewhat pleased to say that many core design elements have turned out the same.
I take that to mean these design decision probably make sense.

It does raise the question as to where the proper line between pandas and numpy lies.
I would argue that evidently, most of pandas functionality has no place in numpy.
Then how is grouping different? I feel what lies at the heart of pandas is a permanent conceptual
association between various pieces of data, assembled in a dataframe and all its metadata.
I think numpy ought to stay well clear of that.
On the other hand, you dont want want to go around creating a pandas dataframe
just to plot a radial reduction; These kind of transient single-statement associations
between keys and values are very useful, entirely independently of some heavyweight framework

Further questions raised from pandas: should we have some form of merge/join functionality too?
or is this getting too panda-ey? all the use cases i can think of fail to be pressed
in some kind of standard mould, but i might be missing something here
"""


import numpy as np
import itertools






"""
this toggle switches between preferred or backwards compatible semantics
for dealing with key objects. current behavior for the arguments to functions
like np.unique is to flatten any input arrays.
i think a more unified semantics is achieved by interpreting all key arguments as
sequences of key objects, whereby non-flat arrays are simply sequences of keys,
whereby the keys themselves are ndim-1-arrays
for reasons of backwards compatibility, it is probably wise to retain the default,
but at least an axis keyword to toggle this behavior would be a welcome addition
"""
backwards_compatible = False
if backwards_compatible:
    axis_default = None
else:
    axis_default = 0



"""
some utility functions
"""

def as_struct_array(*cols):
    """pack a bunch of columns as a struct"""
    cols = [np.asarray(c) for c in cols]
    rows = len(cols[0])
    data = np.empty(rows, [('f'+str(i), c.dtype, c.shape[1:]) for i,c in enumerate(cols)])
    for i,c in enumerate(cols):
        data['f'+str(i)] = c
    return data

def axis_as_object(arr, axis=-1):
    """
    cast the given axis of an array to a void object
    if the axis to be cast is contiguous, a view is returned, otherwise a copy
    this is useful for efficiently sorting by the content of an axis, for instance
    """
    shape = arr.shape
    arr = np.ascontiguousarray(np.swapaxes(arr, axis, -1))
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * shape[axis]))).reshape(np.delete(shape, axis))
def object_as_axis(arr, dtype, axis=-1):
    """cast an array of void objects to a typed axis"""
    return np.swapaxes(arr.view(dtype).reshape(arr.shape+(-1,)), axis, -1)


def array_as_object(arr):
    """view everything but the first axis as a void object"""
    arr = arr.reshape(len(arr),-1)
    return axis_as_object(arr)
def array_as_typed(arr, dtype, shape):
    """unwrap a void object to its original type and shape"""
    return object_as_axis(arr, dtype).reshape(shape)

##def array_as_struct(arr):
##    return np.ascontiguousarray(arr).view([('f0', arr.dtype, arr.shape[1:])])#.flatten()
##def struct_as_array(arr):
##    return arr.view(arr['f0'].dtype)





"""
class hierarchy for indexing a set of keys
the class hierarchy allows for code reuse, while providing specializations for different
types of key objects, yielding maximum flexibility and performance
"""

"""
A note on naming: 'Index' here refers to the fact that the goal of these classes is to
perform and store precomputations on a set of keys,
such as to accelerate subsequent operations involving these keys.
They are not 'logical' indexes as in pandas;
they are not permanently associated with any other data objects

Note that these classes are not primarily intended to be used directly from the numpy namespace,
but rather are intended for code reuse within a family of higher level operations,
only the latter need to be part of the numpy API.

That said, these classes can also be very useful
in those places where the standard operations do not quite cover your needs,
saving your from completely reinventing the wheel.
"""


class BaseIndex(object):
    """
    minimal indexing functionality
    only provides unique and counts, but with optimal performance
    no grouping, or lex-keys are supported,
    or anything that would require an indirect sort
    """
    def __init__(self, keys):
        """
        keys is a flat array of possibly compsite type
        """
        self.keys   = np.asarray(keys).flatten()
        self.sorted = np.sort(self.keys)
        #the slicing points of the bins to reduce over
        self.flag   = self.sorted[:-1] != self.sorted[1:]
        self.slices = np.concatenate((
            [0],
            np.flatnonzero(self.flag)+1,
            [self.size]))

    @property
    def start(self):
        """start index of all bins"""
        return self.slices[:-1]
    @property
    def stop(self):
        """stop index of all bins"""
        return self.slices[1:]
    @property
    def groups(self):
        return len(self.start)
    @property
    def size(self):
        return self.keys.size


    @property
    def unique(self):
        """the first entry of each bin is a unique key"""
        return self.sorted[self.start]
    @property
    def count(self):
        """number of times each key occurs"""
        return np.diff(self.slices)
    @property
    def uniform(self):
        """returns true if each key occurs an equal number of times"""
        return not np.any(np.diff(self.count))


class Index(BaseIndex):
    """
    index object over a set of keys
    adds support for more extensive functionality, notably grouping
    """
    def __init__(self, keys):
        """
        keys is a flat array of possibly composite type
        """
        self.keys   = np.asarray(keys)
        #find indices which sort the keys
        self.sorter = np.argsort(self.keys)
        #computed sorted keys
        self.sorted = self.keys[self.sorter]
        #the slicing points of the bins to reduce over
        self.flag   = self.sorted[:-1] != self.sorted[1:]
        self.slices = np.concatenate((
            [0],
            np.flatnonzero(self.flag)+1,
            [self.size]))

    @property
    def index(self):
        """ive never run into any use cases for this;
        perhaps it was intended to be used to do group_by(keys).first(values)?
        in any case, included for backwards compatibility with np.unique"""
        return self.sorter[self.start]
    @property
    def rank(self):
        """how high in sorted list each key is"""
        return self.sorter.argsort()
    @property
    def sorted_group_rank_per_key(self):
        """find a better name for this? enumeration of sorted keys. also used in median implementation"""
        return np.cumsum(np.concatenate(([False], self.flag)))
    @property
    def inverse(self):
        """return index array that maps unique values back to original space"""
        return self.sorted_group_rank_per_key[self.rank]


##    def where(self, other):
##        """
##        determine at which indices a second set equals a first
##        """
##        return np.searchsorted(self.unique, other)



class ObjectIndex(Index):
    """
    given axis enumerates the keys
    all other axes form the keys
    groups will be formed on the basis of bitwise equality

    should we retire objectindex?
    this can be integrated with regular index ala lexsort, no?
    not sure what is more readable though
    """
    def __init__(self, keys, axis):
        self.axis = axis
        self.dtype = keys.dtype

        keys = np.swapaxes(keys, axis, 0)
        self.shape = keys.shape
        keys = array_as_object(keys)

        super(ObjectIndex, self).__init__(keys)

    @property
    def unique(self):
        """the first entry of each bin is a unique key"""
        sorted = array_as_typed(self.sorted, self.dtype, self.shape)
        return np.swapaxes(sorted[self.start], self.axis, 0)


class LexIndex(Index):
    """
    index object based on lexographic ordering of a tuple of key-arrays
    key arrays can be any type, including multi-dimensional, structed or voidobjects
    however, passing such fancy keys to lexindex may not be ideal from a performance perspective,
    as lexsort does not accept them as arguments directly, so we have to index and invert them first

    should you find yourself with such complex keys, it may be more efficient
    to place them into a structured array first

    note that multidimensional columns are indexed by their first column,
    and no per-column axis keyword is supplied,
    customization of column layout will have to be done at the call site

    """
    def __init__(self, keys):
        self.keys   = tuple(np.asarray(key) for key in keys)
        self.dtypes = tuple(key.dtype for key in self.keys)
        self.shapes = tuple(key.shape for key in self.keys)

        keyviews   = tuple(array_as_object(key) if key.ndim>1 else key for key in self.keys)
        #find indices which sort the keys; complex keys which lexsort does not accept are bootstrapped from Index
        self.sorter = np.lexsort(tuple(Index(key).inverse if key.dtype.kind is 'V' else key for key in keyviews))
        #computed sorted keys
        self.sorted = tuple(key[self.sorter] for key in keyviews)
        #the slicing points of the bins to reduce over
        self.flag   = reduce(
            np.logical_or,
            (s[:-1] != s[1:] for s in self.sorted))
        self.slices = np.concatenate((
            [0],
            np.flatnonzero(self.flag)+1,
            [self.size]))

    @property
    def unique(self):
        """returns a tuple of unique key columns"""
        return tuple(
            (array_as_typed(s, dtype, shape) if len(shape)>1 else s)[self.start]
                for s, dtype, shape in zip(self.sorted, self.dtypes, self.shapes))

    @property
    def size(self):
        return self.sorter.size

class LexIndexSimple(Index):
    """
    simplified LexIndex, which only accepts 1-d arrays of simple dtypes
    the more expressive LexIndex only has some logic overhead,
    in case all columns are indeed simple. not sure this is worth the extra code
    """
    def __init__(self, keys):
        self.keys   = tuple(np.asarray(key) for key in keys)
        self.sorter = np.lexsort(self.keys)
        #computed sorted keys
        self.sorted = tuple(key[self.sorter] for key in self.keys)
        #the slicing points of the bins to reduce over
        self.flag   = reduce(
            np.logical_or,
            (s[:-1] != s[1:] for s in self.sorted))
        self.slices = np.concatenate((
            [0],
            np.flatnonzero(self.flag)+1,
            [self.size]))

    @property
    def unique(self):
        """the first entry of each bin is a unique key"""
        return tuple(s[self.start] for s in self.sorted)

    @property
    def size(self):
        return self.sorter.size



def as_index(keys, axis = axis_default, base=False):
    """
    casting rules for a keys object to an index

    the preferred semantics is that keys is a sequence of key objects,
    except when keys is an instance of tuple,
    in which case the zipped elements of the tuple are the key objects

    the axis keyword specifies the axis which enumerates the keys
    if axis is None, the keys array is flattened
    if axis is 0, the first axis enumerates the keys
    which of these two is the default depends on whether backwards_compatible == True

    if base==True, the most basic index possible is constructed.
    this avoids an indirect sort; if it isnt required, this has better performance
    """
    if isinstance(keys, Index):
        return keys
    if isinstance(keys, tuple):
        return LexIndex(keys)
    try:
        keys = np.asarray(keys)
    except:
        raise TypeError('Given object does not form a valid set of keys')
    if axis is None:
        keys = keys.flatten()
    if keys.ndim==1:
        if base:
            return BaseIndex(keys)
        else:
            return Index(keys)
    else:
        return ObjectIndex(keys, axis)





"""
public API starts here
"""


class GroupBy(object):
    """
    supports ufunc reduction
    should any other form of reduction be supported?
    not sure it should; more cleanly written externally i think, on a grouped iterables
    """
    def __init__(self, keys, axis = 0):
        #we could inherit from Index, but the multiple specializations make
        #holding a reference to an index object more appropriate
        # note that we dont have backwards compatibility issues with groupby,
        #so we are going to use axis = 0 as a default
        #the multi-dimensional structure of a keys object is usualy meaningfull,
        #and value arguments are not automatically flattened either

        self.index = as_index(keys, axis)

    #forward interesting/'public' index properties
    @property
    def unique(self):
        return self.index.unique
    @property
    def count(self):
        return self.index.count
    @property
    def inverse(self):
        return self.index.inverse
    @property
    def rank(self):
        return self.index.rank


    #some different methods of chopping up a set of values by key
    #not sure they are all equally relevant, but i actually have real world use cases for most of them

    def as_iterable_from_iterable(self, values):
        """
        grouping of an iterable. memory consumption depends on the amount of sorting required
        worst case, if index.sorter[-1] = 0, we need to consume the entire value iterable,
        before we can start yielding any output
        but to the extent that the keys come presorted, the grouping is lazy
        """
        values = iter(enumerate(values))
        cache = dict()
        def get_value(ti):
            try:
                return cache.pop(ti)
            except:
                while True:
                    i, v = next(values)
                    if i==ti:
                        return v
                    cache[i] = v
        s = iter(self.index.sorter)
        for c in self.count:
            yield (get_value(i) for i in itertools.islice(s, c))

    def as_outoforder(self, values):
        """
        group values, without regard for the ordering of self.index.unique
        consume values as they come, and yield key-group pairs as soon as they complete
        thi spproach is lazy, insofar as grouped values are close in their iterable
        """
        from collections import defaultdict
        cache = defaultdict(list)
        count = self.count
        unique = self.unique
        key = (lambda i: unique[i]) if isinstance(unique, np.ndarray) else (lambda i: tuple(c[i] for c in unique))
        for i,v in itertools.izip(self.inverse, values):
            cache[i].append(v)
            if len(cache[i]) == count[i]:
                yield key(i), cache.pop(i)

    def as_iterable_from_sequence(self, values):
        """
        this is the preferred method if values has random access,
        but we dont want it completely in memory.
        like a big memory mapped file, for instance
        """
        s = iter(self.index.sorter)
        for c in self.count:
            yield (values[i] for i in itertools.islice(s, c))

    def as_array(self, values):
        """
        return grouped values as an ndarray
        returns an array of shape [groups, groupsize, ungrouped-axes]
        this is only possible if index.uniform==True
        """
        assert(self.index.uniform)
        values = np.asarray(values)
        values = values[self.index.sorter]
        return values.reshape(self.index.groups, self.count[0], *values.shape[1:])

    def as_list(self, values):
        """return grouped values as a list of arrays, or a jagged-array"""
        values = np.asarray(values)
        values = values[self.index.sorter]
        return np.split(values, self.index.slices[1:-1], axis=0)

    def group(self, values):
        try:
            return self.as_array(values)
        except:
            return self.as_list(values)

    def __call__(self, values):
        """
        not sure how i feel about this. explicit is better than implict
        but group_by(keys).group(values) does not sit too well with me either
        """
        return self.unique, self.group(values)


    # ufunc based reduction methods

    def reduce(self, values, operator = np.add):
        """
        reduce the values over identical key groups, using the ufunc operator
        reduction is over the first axis, which should have elements corresponding to the keys
        all other axes are treated indepenently for the sake of this reduction
        note that this code is only C-vectorized over the first axis
        that is fine is this inner loop is significant, but not so much if it isnt
        if we have only few keys, but values[0].size is substantial, a reduction via
        as_list may be preferable
        """
        values = values[self.index.sorter]
        if values.ndim>1:
            return np.apply_along_axis(
                lambda slc: operator.reduceat(slc, self.index.start),
                0, values)
        else:
            return operator.reduceat(values, self.index.start)

    def sum(self, values, axis=0):
        """compute the sum over each group"""
        values = np.asarray(values)
        if axis: values = np.rollaxis(values, axis)
        return self.unique, self.reduce(values)

    def mean(self, values, axis=0):
        """compute the mean over each group"""
        values = np.asarray(values)
        if axis: values = np.rollaxis(values, axis)
        count = self.count.reshape(-1,*(1,)*(values.ndim-1))
        return self.unique, self.reduce(values) / count

    def var(self, values, axis=0):
        """compute the variance over each group"""
        values = np.asarray(values)
        if axis: values = np.rollaxis(values, axis)
        count = self.count.reshape(-1,*(1,)*(values.ndim-1))
        mean = self.reduce(values) / count
        err = values - mean[self.inverse]
        return self.unique, self.reduce(err**2) / count

    def std(self, values, axis=0):
        """standard deviation over each group"""
        unique, var = self.var(values, axis)
        return unique, np.sqrt(var)

    def median(self, values, axis=0, average=True):
        """
        compute the median value over each group.
        when average is true, the average is the two cental values is taken
        for groups with an even key-count
        """
        values = np.asarray(values)

        mid_2 = self.index.start + self.index.stop
        hi = (mid_2    ) // 2
        lo = (mid_2 - 1) // 2

        #need this indirection for lex-index compatibility
        sorted_group_rank_per_key = self.index.sorted_group_rank_per_key

        def median1d(slc):
            #place values at correct keys; preconditions the upcoming lexsort
            slc    = slc[self.index.sorter]
            #refine value sorting within each keygroup
            sorter = np.lexsort((slc, sorted_group_rank_per_key))
            slc    = slc[sorter]
            return (slc[lo]+slc[hi]) / 2 if average else slc[hi]

        values = np.asarray(values)
        if axis: values = np.rollaxis(values, axis)
        if values.ndim>1:   #is trying to skip apply_along_axis somewhat premature optimization?
            values = np.apply_along_axis(median1d, 0, values)
        else:
            values = median1d(values)
        return self.unique, values

    def min(self, values, axis=0):
        """return the minimum within each group"""
        values = np.asarray(values)
        if axis: values = np.rollaxis(values, axis)
        return self.unique, self.reduce(values, np.minimum)

    def max(self, values, axis=0):
        """return the maximum within each group"""
        values = np.asarray(values)
        if axis: values = np.rollaxis(values, axis)
        return self.unique, self.reduce(values, np.maximum)

    def first(self, values, axis=0):
        """return values at first occurance of its associated key"""
        values = np.asarray(values)
        if axis: values = np.rollaxis(values, axis)
        return self.unique, values[self.index.sorter[self.index.start]]

    def last(self, values, axis=0):
        """return values at last occurance of its associated key"""
        values = np.asarray(values)
        if axis: values = np.rollaxis(values, axis)
        return self.unique, values[self.index.sorter[self.index.stop-1]]



    #implement iter interface? could simply do zip( group_by(keys)(values)), no?

#just an alias, for those who dont like camelcase
group_by = GroupBy
#could also turn this into a function with optional values and reduction func.
def group_by(keys, values=None, reduction=None, axis=0):
    g = GroupBy(keys, axis)
    if values is None:
        return g
    groups = g.group(values)
    if reduction is None:
        return groups
    return [reduction(group) for group in groups]



"""
some demonstration of the duplicity between this code and np.unique
they share a lot of common functionality
we cant quite bootstrap grouping from unique as is
but unique can easily be reimplemented using the Index class
"""

def unique(keys, return_index = False, return_inverse = False, return_count = False, axis = axis_default):
    """
    backwards compatible interface with numpy.unique
    in the long term i think the kwargs should be deprecated though
    cleaner to call index and its properties directly,
    should you want something beyond the interface provided
    """
    index = as_index(keys, axis, base = not (return_index or return_inverse))

    ret = index.unique,
    if return_index:
        ret = ret + (index.index,)
    if return_inverse:
        ret = ret + (index.inverse,)
    if return_count:
        ret = ret + (index.count,)
    return ret[0] if len(ret) == 1 else ret

def count(keys, axis = axis_default):
    """numpy work-alike of collections.Counter"""
    index = as_index(keys, axis, base = True)
    return index.unique, index.count

def multiplicity(keys, axis = axis_default):
    """
    return the multiplicity of each key, or how often it occurs in the set
    note that we are not interested in the unique values for this operation,
    casting doubt on the current numpy design which places np.unique
    at the center of the interface for this kind of operation
    given how often i use multiplicity, id like to have it in the numpy namespace
    """
    index = as_index(keys, axis)
    return index.count[index.inverse]

def rank(keys, axis = axis_default):
    """
    where each item is in the pecking order.
    not sure this should be part of the public api, cant think of any use-case right away
    plus, we have a namespace conflict
    """
    index = as_index(keys, axis)
    return index.rank


def incidence(boundary):
    """
    given an Nxm matrix containing boundary info between simplices,
    compute indidence info matrix
    not to be part of numpy API, to be sure, just something im playing with
    """
    return group_by(boundary).group(np.arange(boundary.size) // boundary.shape[1])




def test_basic():

    keys   = ["e", "b", "b", "c", "d", "e", "e", 'a']
    values = [1.2, 4.5, 4.3, 2.0, 5.67, 8.08, 9.01,1]

##    keys   = np.array(["a", "b", "b", "c", "d", "e", "e",'d','a','c'])
##    values = np.array([1.2, 4.5, 4.3, 2.0, 5.67, 8.08, 9.01, 1,2,3])

    print ('two methods of computing non-reducing group')
    print ('as iterable')
    g = group_by(keys)
    for k,v in zip(g.unique, g.as_iterable_from_sequence(values)):
        print (k, list(v))
    print ('as list')
    for k,v in zip(*group_by(keys)(values)):
        print (k, v)

    print ('some reducing group operations')
    g = group_by(keys)
    unique_keys, reduced_values = g.median(values)
    print ('per group median')
    print (reduced_values)
    unique_keys, reduced_values = g.mean(values)
    print ('per group mean')
    print (reduced_values)
    unique_keys, reduced_values = g.std(values)
    print ('per group std')
    print (reduced_values)
    reduced_values = g.reduce(np.array(values), np.minimum) #alternate way of calling
    print ('per group min')
    print (reduced_values)
    unique_keys, reduced_values = g.max(values)
    print ('per group max')
    print (reduced_values)


##test_basic()
##quit()


def test_lex_median():
    """
    for making sure i squased all bugs related to fancy-keys and median filter implementation
    """
    keys1  = ["e", "b", "b", "c", "d", "e", "e", 'a']
    keys2  = ["b", "b", "b", "d", "e", "e", 'e', 'e']
##    keys3 = np.random.randint(0,2,(8,2))
    values = [1.2, 4.5, 4.3, 2.0, 5.6, 8.8, 9.1, 1]

    unique, median = group_by((keys1, keys2)).median(values)
    for i in zip(zip(*unique), median):
        print (i)

##test_lex_median()
##quit()


def test_fancy_keys():
    """
    test Index subclasses
    """
    keys        = np.random.randint(0, 2, (20,3)).astype(np.int8)
    values      = np.random.randint(-1,2,(20,4))


    #all these various datastructures should produce the same behavior
    #multiplicity is a nice unit test, since it draws on most of the low level functionality
    if backwards_compatible:
        assert(np.all(
            multiplicity(keys, axis=0) ==           #void object indexing
            multiplicity(tuple(keys.T))))           #lexographic indexing
        assert(np.all(
            multiplicity(keys, axis=0) ==           #void object indexing
            multiplicity(as_struct_array(keys))))   #struct array indexing
    else:
        assert(np.all(
            multiplicity(keys) ==                   #void object indexing
            multiplicity(tuple(keys.T))))           #lexographic indexing
        assert(np.all(
            multiplicity(keys) ==                   #void object indexing
            multiplicity(as_struct_array(keys))))   #struct array indexing

    #lets go mixing some dtypes!
    floatkeys   = np.zeros(len(keys))
    floatkeys[0] = 8.8
    print ('sum per group of identical rows using struct key')
    g = group_by(as_struct_array(keys, floatkeys))
    for e in zip(g.count, *g.sum(values)):
        print (e)
    print ('sum per group of identical rows using lex of nd-key')
    g = group_by(( keys, floatkeys))
    for e in zip(zip(*g.unique), g.sum(values)[1]):
        print (e)
    print ('sum per group of identical rows using lex of struct key')
    g = group_by((as_struct_array( keys), floatkeys))
    for e in zip(zip(*g.unique), g.sum(values)[1]):
        print (e)

    #showcase enhanced unique functionality
    images = np.random.rand(4,4,4)
    #shuffle the images; this is a giant mess now; how to find the unique ones?
    shuffled = images[np.random.randint(0,4,200)]
    #there you go
    if backwards_compatible:
        print (unique(shuffled, axis=0))
    else:
        print (unique(shuffled))


##g = test_fancy_keys()
##quit()


def test_radial():
    x = np.linspace(-2,2, 64)
    y = x[:, None]
    x = x[None, :]
    R = np.sqrt( x**2+y**2)

    def airy(r, sigma):
        from scipy.special import j1
        r = r / sigma * np.sqrt(2)
        a = (2*j1(r)/r)**2
        a[r==0] = 1
        return a
    def gauss(r, sigma):
        return np.exp(-(r/sigma)**2)

    distribution = np.random.choice([gauss, airy])(R, 0.3)
    sample = np.random.poisson(distribution*200+10).astype(np.float)

    import matplotlib.pyplot as pp
    #is this an airy or gaussian function? hard to tell with all this noise!
    pp.imshow(sample, interpolation='nearest', cmap='gray')
    pp.show()
    #radial reduction to the rescue!
    #if we are sampling an airy function, you will see a small but significant rise around x=1
    g = group_by(np.round(R, 5).flatten())
    pp.errorbar(
        g.unique,
        g.mean(sample.flatten())[1],
        g.std (sample.flatten())[1] / np.sqrt(g.count))
    pp.xlim(0,2)
    pp.show()

##test_radial()
##quit()

def test_meshing():
    """
    meshing example
    demonstrates the use of multiplicity, and group.median
    """
    #set up some random points, and get their delaunay triangulation
    points = np.random.random((20000,2))*2-1
    points = points[np.linalg.norm(points,axis=1) < 1]
    from scipy.spatial.qhull import Delaunay
    d = Delaunay(points)
    tris = d.simplices

    #the operations provided in this module allow us to express potentially complex
    #computational geometry questions elegantly in numpy
    #Delaunay.neighbors could be used as well,
    #but the point is to express this in pure numpy, without additional library functionaoty
    edges = tris[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    sorted_edges = np.where(edges[:,0:1]<edges[:,1:2], edges, edges[:,::-1])
    #we test to see how often each edge occurs, or how many indicent simplices it has
    #this is a very general method of finding the boundary of any topology
    #and we can do so here with only one simple and readable command, multiplicity == 1
    if backwards_compatible:
        boundary_edges = edges[multiplicity(sorted_edges, axis=0)==1]
    else:
        boundary_edges = edges[multiplicity(sorted_edges)==1]
    boundary_points = unique(boundary_edges)

    if False:
        print (boundary_edges)
        print (incidence(boundary_edges))


    #create some random values on faces
    #we want to smooth them over the mesh to create a nice hilly landscape
    face_values   = np.random.normal(size=d.nsimplex)
    #add some salt and pepper noise, to make our problem more interesting
    face_values[np.random.randint(d.nsimplex, size=10)] += 1000

    #start with a median step, to remove salt-and-pepper noise
    #toggle to mean to see the effect of the median filter
    g = group_by(tris.flatten())
    prestep = g.median if True else g.mean
    vertex_values = prestep(np.repeat(face_values, 3))[1]
    vertex_values[boundary_points] = 0

    #actually, we can compute the mean without grouping
    tris_per_vert = g.count
    def scatter(x):
        r = np.zeros(d.npoints, x.dtype)
        for idx in tris.T: np.add.at(r, idx, x)
        return r / tris_per_vert
    def gather(x):
        return x[tris].mean(axis=1)

    #iterate a little
    for i in range(100):
        face_values   = gather(vertex_values)
        vertex_values = scatter(face_values)
        vertex_values[boundary_points] = 0


    #display our nicely rolling hills and their boundary
    import matplotlib.pyplot as plt
    x, y = points.T
    plt.tripcolor(x,y, triangles = tris, facecolors = face_values)
    plt.scatter(x[boundary_points], y[boundary_points])
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    # d = test_meshing()
    test_basic()
