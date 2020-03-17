import re
import ast, astor
from ast import Attribute, Name

code_path = 'code.py'
api_name = ['c_', 'iloc',  'loc', 'apply','sum', 'count_nonzero','hstack','array', 'where', 'transpose', 'query',
            'vstack','zeros','query','apply','map','crosstab','where','atleast_2d','tile','loc',
            'to_datetime','array','hstack','concatenate','c_','iloc','astype','array','str',
            'fillna','norm','where','array','column_stack','count_nonzero','nonzero','transpose',
            'replace','cumprod','tensordot','iterrows','full','query','arange',
            'array','column_stack','dot','full','hstack','ones','vstack','zeros','ix','dot']


class CallParser(ast.NodeVisitor):
    def __init__(self):
        self.attrs = []
        self.name = ""


    def generic_visit(self, node):
        rlist = list(ast.iter_fields(node))
        for field, value in reversed(rlist):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)


    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, Attribute) and node.func.attr in api_name:
                self.attrs.append(node)
        elif isinstance(node.func, Name) and node.func.id in api_name:
                self.attrs.append(node)


    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.value, Attribute) and node.value.attr in api_name:
            self.attrs.append(node)


# insert new code
class CodeInstrumentator(ast.NodeTransformer):
    def __init__(self, lineno, newnode):
        self.line = lineno
        self.newnode = newnode

    def generic_visit(self, node):
        rlist = list(ast.iter_fields(node))
        for field, value in reversed(rlist):
            if isinstance(value, list):
                for item in value:
                    if hasattr(item, 'lineno') and item.lineno == self.line:
                        index = value.index(item)
                        value.insert(index + 1, self.newnode)
                        return node
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)


# 直接替换的情况
def replace_one(node):

    oldstmt = astor.to_source(node).strip()

    #   pd.Series.str ->pd.Series.map
    if '.str' in oldstmt:
        newstmt = oldstmt.replace('str', 'map(lambda x: x') + ')'
        return newstmt
    #  pd.Series.astype -> pd.Series.apply
    elif '.astype' in oldstmt:
        newstmt = oldstmt.replace('astype','apply')
        return newstmt
    # np.hstack -> np.concatenate
    elif '.arrange' in oldstmt:
        newstmt = oldstmt.replace('arrange(', 'np.array(range(')
        return newstmt
    # pd.DataFrame.loc ->pd.DataFrame.ix
    elif '.loc' in oldstmt:
        newstmt = oldstmt.replace('loc', 'ix')
        return newstmt
    # pd.DataFrame.ix -> pd.DataFrame.loc
    elif '.ix' in oldstmt:
        newstmt = oldstmt.replace('ix', 'loc')
        return newstmt
    # np.ones -> np.empty, np.ndarray.fill
    elif '.ones' in oldstmt:
        newstmt = oldstmt.replace("np.ones", "np.empty") + "; r2.fill(1)"
        return newstmt
    # np.zeros -> np.empty
    elif '.zeros'in oldstmt:
        newstmt = oldstmt.replace("zeros", "empty") + "; r2[:]= 0"
        return newstmt
    # np.nonzero -> np.where
    elif '.nonzero' in oldstmt:
        newstmt = oldstmt.replace("nonzero", "where")
        return newstmt
    # pd.DataFrame.query -> pd.DataFrame.loc
    elif '.query' in oldstmt:
        newstmt = oldstmt.replace('query', 'loc')
        return newstmt
    # np.column_stack	np.transpose
    elif '.column_stack' in oldstmt:
        newstmt = oldstmt.replace("column_stack", "transpose")
        return newstmt
    # np.transpose->np.column_stack
    elif '.transpose' in oldstmt:
        newstmt = oldstmt.replace("transpose", "column_stack")
        return newstmt
    # np.where -> np.nonzero
    elif '.where' in oldstmt:
        newstmt = oldstmt.replace("where", "nonzero")
        return newstmt
    # np.c_ -> np.hstack
    elif '.c_' in oldstmt:
        newstmt = oldstmt.replace("c_", "hstack(") + ")"
        return newstmt

    # pd.Series.apply -> pd.Series.map
    elif '.apply' in oldstmt:
        newstmt = oldstmt.replace('apply', 'map')
        return newstmt
    # pd.Series.replace	-> pd.Series.map
    elif '.replace' in oldstmt:
        newstmt = oldstmt.replace('replace', 'map')
        return newstmt

    # pd.Series.iloc ->pd.Series.iat
    elif '.iloc' in oldstmt:
        newstmt = oldstmt.replace('iloc', 'iat')
        return newstmt


# 可以直接替换，参数要换一下
def replace_two(node):
    oldstmt = astor.to_source(node).strip()
    # np.column_stack -> np.vstack
    if '.column_stack' in oldstmt:
        string = re.match('np.column_stack\([\(\[](.*)[\)\]]\)$', oldstmt).group(1)
        args = string.split(',')
        args = map(lambda arg: '%s%s' % (arg, '.T'), args)
        args = ','.join(list(args))
        newstmt = 'np.vstack((' + args + ')).T'
        return newstmt
    # np.vstack ->	np.column_stack
    elif '.vstack' in oldstmt:
        string = re.match('np.vstack\([\(\[](.*)[\)\]]\)$', oldstmt).group(1)
        args = string.split(',')
        args = map(lambda arg: '%s%s' % (arg, '.T'), args)
        args = ','.join(list(args))
        newstmt = 'np.column_stack((' + args + ')).T'
        return newstmt
    # np.count_nonzero -> np.ndarray.sum
    elif '.count_nonzero' in oldstmt:
        agr = re.match('np\.count_nonzero\((.*)\)$', oldstmt).group(1)
        newstmt = 'np.sum(' + agr + '!=0)'
        return newstmt
    # np.full -> np.empty
    elif '.full' in oldstmt:
        oldstmt = oldstmt.replace('\n', '\t')
        agrument1 = re.match('np.full\((.*)\,(.*)\,(.*)\)$', oldstmt).group(1)
        agrument2 = re.match('np.full\((.*)\,(.*)\,(.*)\)$', oldstmt).group(2)
        agrument3 = re.match('np.full\((.*)\,(.*)\,(.*)\)$', oldstmt).group(3)
        newstmt = 'np.empty(' + agrument1 + ',' + agrument3 + ') ; r2[:] =' + agrument2
        return newstmt
    # np.sum ->np.ndarray.sum
    elif '.sum' in oldstmt:
        if re.match('np\.sum\(.*\,.*\)$', oldstmt):
            agrument1 = re.match('np\.sum\((.*)\,(.*)\)$', oldstmt).group(1)
            agrument2 = re.match('np\.sum\((.*)\,(.*)\)$', oldstmt).group(2)
        elif re.match('np\.sum\((.*)\)$', oldstmt):
            agrument1 = re.match('np\.sum\((.*)\)$', oldstmt).group(1)
            agrument2 = ""
        newstmt = '(' + agrument1 + ').sum(' + agrument2 + ')'
        return newstmt
    # np.ndarray.dot -> np.tensordot
    elif '.dot' in oldstmt:
        if 'np.dot.*' not in oldstmt:
            agr1 = re.match('(.*)\.dot\((.*)\)$', oldstmt).group(1)
            agr2 = re.match('(.*)\.dot\((.*)\)$', oldstmt).group(2)
            newstmt = 'np.tensordot(' + agr1 + ',' + agr2 + ',axes=1)'
            return newstmt



# 需要在特定的使用模式下才能替换,但参数不需要换
def replace_three(node):
    oldstmt = astor.to_source(node).strip()
    # np.array ->np.fromiter
    if re.match('np.array\(.*dtype.*\)$', oldstmt):
        newstmt = oldstmt.replace('array', 'fromiter')
        return newstmt
    # np.array ->np.hstack
    elif re.match('np\.array\([^\,]*\)$', oldstmt):
        newstmt = oldstmt.replace('array', 'hstack(') + ')'
        return newstmt
    # np.linalg.norm ->np.ndarray.sum, np.sqrt
    elif re.match('np\.linalg\.norm\(.*\)$', oldstmt):
        newstmt = oldstmt.replace('np.linalg.norm', 'np.sqrt(np.square') + '.sum())'
        return newstmt
    # np.sum ->np.count_nonzero
    elif re.match('np\.sum\(.*==.*\)$', oldstmt):
        newstmt = oldstmt.replace('sum', 'count_nonzero')
        return newstmt
    # pd.Series.map -> pd.Series.replace && pd.Series.map -> pd.DataFrame.replace
    """elif re.match('.*map.*lambda.*', oldstmt) is None:
        newstmt = oldstmt.replace('map', 'replace')
        return newstmt"""


# 需要在特定使用模式下才能替换，并且参数需要转换
def replace_four(node):
    oldstmt = astor.to_source(node).strip()
    # np.where->pd.DataFrame.apply
    if re.match('np.where\(.*\,.*\,.*\)$', oldstmt):
        for c in ast.iter_child_nodes(node):
            if isinstance(c, ast.Compare):
                o = astor.to_source(c.left).strip()
                condition = re.match('np.where\((.*)\,(.*)\,(.*)\)$', oldstmt).group(1)
                t = re.match('np.where\((.*)\,(.*)\,(.*)\)$', oldstmt).group(2)
                f = re.match('np.where\((.*)\,(.*)\,(.*)\)$', oldstmt).group(3)
                newstmt = 'pd.DataFrame('+o+').apply(lambda x : ' + condition.replace(o, 'x') + ').replace({True:'+t+',False:'+f+'}).values.flatten()'
                return newstmt

    # np.where->pd.Series.map
    elif re.match('np.where\(.*\,.*\,.*\)$', oldstmt):
        for c in ast.iter_child_nodes(node):
            if isinstance(c, ast.Compare):
                o = astor.to_source(c.left).strip()
                condition = re.match('np.where\((.*)\,(.*)\,(.*)\)$', oldstmt).group(1)
                t = re.match('np.where\((.*)\,(.*)\,(.*)\)$', oldstmt).group(2)
                f = re.match('np.where\((.*)\,(.*)\,(.*)\)$', oldstmt).group(3)
                newstmt = '(' + o + ').map(lambda x :' + t + ' if ' + condition.replace(o, 'x') + ' else ' + f + ').values'
                return newstmt

    # np.hstack -> np.concatenate
    elif re.match('np.hstack.*', oldstmt):
        for node in ast.iter_child_nodes(candidate):
            if not isinstance(node, ast.Attribute):
                agr = astor.to_source(node).strip()
                newstmt = 'np.concatenate(' + agr + ', axis=1)'
                return newstmt

    # np.vstack -> np.concatenate
    elif re.match('np.vstack.*', oldstmt):
        for node in ast.iter_child_nodes(candidate):
            if not isinstance(node, ast.Attribute):
                agr = astor.to_source(node).strip()
                newstmt = 'np.concatenate(' + agr + ', axis=0)'
                return newstmt

    # np.array -> np.concatenate
    elif re.match('np\.array\([^\,]*\)$', oldstmt):
        o = re.match('np\.array\(([^\,]*)\)$', oldstmt).group(1)
        newstmt = 'np.concatenate([' + o + '])'
        return newstmt

    # pd.DataFrame.fillna -> pd.DataFrame.combine_first
    elif re.match('.*\.fillna\(.*inplace.*\)$', oldstmt):
        o = re.match('(.*)\.fillna\((.*)\)', oldstmt).group(1)
        arg = re.match('(.*)\.fillna\((.*)\)', oldstmt).group(2)
        newstmt = o + '.combine_first(' + o + '.applymap(lambda x: ' + arg + '))'
        return newstmt

    # np.dot -> np.einsum
    elif re.match('np.dot.*', oldstmt):
        agr1 = re.match('np.dot\((.*)\)$', oldstmt).group(1)
        newstmt = "np.einsum('ij,jm->im'," + agr1 + ')'
        return newstmt

    # np.sum ->np.einsum
    elif re.match('np\.sum\(.*\,.*\)$', oldstmt):
        agrument1 = re.match('np\.sum\((.*)\,(.*)\)$', oldstmt).group(1)
        agrument2 = re.match('np\.sum\((.*)\,(.*)\)$', oldstmt).group(2)
        if ('axis=1' in agrument2) or ('axis=-1' in agrument2):
            newstmt = "np.einsum('ij->i'," + agrument1 + ")"
            print(newstmt)
        elif ('axis=-2' in agrument2) or ('axis=0' in agrument2):
            newstmt = "np.einsum('ij->j'," + agrument1 + ")"
        return newstmt
    # np.sum ->np.einsum
    elif re.match('np\.sum\((.*)\)$', oldstmt):
        agrument1 = re.match('np\.sum\((.*)\)$', oldstmt).group(1)
        newstmt = "np.einsum('i->'," + agrument1 + ")"
        return newstmt

    # np.ndarray.sum - >p.einsum
    elif re.match('(.*)\.sum\((.*)\)$', oldstmt):
        agrument1 = re.match('(.*)\.sum\((.*)\)$', oldstmt).group(1)
        agrument2 = re.match('(.*)\.sum\((.*)\)$', oldstmt).group(2)
        if ('axis=1' in agrument2) or ('axis=-1' in agrument2):
            newstmt = "np.einsum('ij->i'," + agrument1 + ")"
        elif ('axis=-2' in agrument2) or ('axis=0' in agrument2):
            newstmt = "np.einsum('ij->j'," + agrument1 + ")"
        else:
            newstmt = "np.einsum('i->'," + agrument1 + ")"
        return newstmt



    # np.hstack->np.append
    elif re.match('np.hstack.*', oldstmt):
        for node in ast.iter_child_nodes(candidate):
            if isinstance(node, ast.List) or isinstance(node, ast.Tuple):
                agr = astor.to_source(node).strip()
                agr = re.match('^[\[\(](.*)[\]\)]$', agr).group(1)
            else:
                continue
        newstmt = 'np.append(' + agr + ', axis=1)'
        return newstmt

    # np.where -> np.ndarray.astype
    elif re.match('np\.where\(.*\,\s1\,\s0\)$', oldstmt):
        agrument1 = re.match('np\.where\((.*)\,\s1\,\s0\)$', oldstmt).group(1)
        newstmt = '(' + agrument1 + ').astype((' + agrument1 + ').dtype)'
        return newstmt

    # pd.Series.apply ->pd.DataFrame.apply
    elif re.match('.*[\[].*apply\(lambda.*', oldstmt) and 'axis' not in oldstmt:
        o = re.match('(.*)\.apply\(.*', oldstmt).group(1)
        z = re.match('(.*)\[', o).group(1)
        x = re.match('.*\.apply\(lambda\s(.*):.*\)$', oldstmt).group(1)
        z = o.replace(z, x)
        remain = re.match('.*\.apply\(lambda(.*):(.*)\)$', oldstmt).group(2)
        remain = remain.replace(x, z)
        newstmt = 'pd.DataFrame(' + o + ').apply(lambda ' + x + ':' + remain + ',axis=1)'
        return newstmt

    # pd.Series.map	-> np.where
    elif re.match('.*map.*lambda.*if.*else.*', oldstmt):
        o = re.match('(.*)\.map\(lambda(.*):(.*)if(.*)else\s(.*)\)$', oldstmt).group(1)
        x = re.match('(.*)\.map\(lambda\s(.*):(.*)if(.*)else\s(.*)\)$', oldstmt).group(2)
        t = re.match('(.*)\.map\(lambda(.*):(.*)if(.*)else\s(.*)\)$', oldstmt).group(3)
        condition = re.match('(.*)\.map\(lambda(.*):(.*)if(.*)else\s(.*)\)$', oldstmt).group(4)
        f = re.match('(.*)\.map\(lambda(.*):(.*)if(.*)else\s(.*)\)$', oldstmt).group(5)
        newstmt = 'np.where(' + condition.replace(x, o) + ',' + t.replace(x, o) + ',' + f.replace(x, o) + ')'
        return newstmt
    # pd.DataFrame.apply -> pd.Series.apply
    elif re.match('.*apply.*lambda.*if.*else.*', oldstmt) and 'axis=1' in oldstmt:
        o = re.match('(.*)\.apply\(lambda(.*):(.*)if(.*)else\s(.*)\,.*\)$', oldstmt).group(1)
        x = re.match('(.*)\.apply\(lambda\s(.*):(.*)if(.*)else\s(.*)\,.*\)$', oldstmt).group(2)
        t = re.match('(.*)\.apply\(lambda(.*):(.*)if(.*)else\s(.*)\,.*\)$', oldstmt).group(3)
        f = re.match('(.*)\.apply\(lambda(.*):(.*)if(.*)else\s(.*)\,.*\)$', oldstmt).group(5)
        condition = re.match('(.*)\.apply\(lambda(.*):(.*)if(.*).*else\s(.*)\,.*\)$', oldstmt).group(4)
        z = re.match('(.*)\.apply\(lambda(.*):(.*)if(.*\]).*else\s(.*)\,.*\)$', oldstmt).group(4)
        newstmt = z.replace(x, o) + '.apply(lambda ' + x + ':' + t.replace(z, x) + 'if ' + condition.replace(z,x) + 'else ' + f.replace(z, x) + ')'
        return newstmt
    # pd.DataFrame.apply -> np.where
    elif re.match('.*apply.*lambda.*if.*else.*', oldstmt) and 'axis=1' in oldstmt:
        o = re.match('(.*)\.apply\(lambda(.*):(.*)if(.*)else\s(.*)\,.*\)$', oldstmt).group(1)
        x = re.match('(.*)\.apply\(lambda\s(.*):(.*)if(.*)else\s(.*)\,.*\)$', oldstmt).group(2)
        t = re.match('(.*)\.apply\(lambda(.*):(.*)if(.*)else\s(.*)\,.*\)$', oldstmt).group(3)
        condition = re.match('(.*)\.apply\(lambda(.*):(.*)if(.*)else\s(.*)\,.*\)$', oldstmt).group(4)
        f = re.match('(.*)\.apply\(lambda(.*):(.*)if(.*)else\s(.*)\,.*\)$', oldstmt).group(5)
        newstmt = 'np.where(' + condition.replace(x, o) + ',' + t.replace(x, o) + ',' + f.replace(x, o) + ')'
        return newstmt
    # np.full->np.zeros
    elif re.match('np.full\((.*)\,\s0\,(.*)\)$', oldstmt):
        agrument1 = re.match('np.full\((.*)\,\s0\,(.*)\)$', oldstmt).group(1)
        agrument2 = re.match('np.full\((.*)\,\s0\,(.*)\)$', oldstmt).group(2)
        newstmt = 'np.zeros(' + agrument1 + ',' + agrument2 + ')'
        return newstmt

# 打开这文件
with open(code_path, 'r', encoding='UTF-8') as file:
    content = file.read()
file.close()

# 解析
tree = ast.parse(content)  # parse the code into ast_tree
v = CallParser()   # find the potential api
v.visit(tree)

for candidate in v.attrs:
    oldstmt = astor.to_source(candidate).strip()
    lineno = candidate.lineno
    print("original API:" +oldstmt)
    if replace_one(candidate):
        print('Recommend API:' + replace_one(candidate))
    if replace_two(candidate):
        print('Recommend API:' + replace_two(candidate))
    if replace_three(candidate):
        print('Recommend API:' + replace_three(candidate))
    if replace_four(candidate):
        print('Recommend API:' + replace_four(candidate))

    print("lineno:{}".format(lineno))
    print('###############################################')





