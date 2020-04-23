import re
import ast, astor
from ast import Attribute, Name
import os,csv
import subprocess

api_name = ['c_', 'iloc',  'loc', 'apply','sum', 'count_nonzero','hstack','array', 'where', 'transpose', 'query',
            'vstack','zeros','query','apply','map','crosstab','where','atleast_2d','tile','loc',
            'to_datetime','array','hstack','concatenate','c_','iloc','astype','array','str',
            'fillna','norm','where','array','column_stack','count_nonzero','nonzero','transpose',
            'replace','cumprod','tensordot','iterrows','full','query','arange',
            'array','column_stack','dot','full','hstack','ones','vstack','zeros','ix','dot']

api_pair = {'astype': 'apply','column_stack': 'transpose',
            'loc': 'ix', 'transpose': 'column_stack',
            'ix': 'loc', 'where': 'nonzereo', 'nonzereo': 'where',
            'apply': 'map', 'query': 'loc', 'replace': 'map'}
# template for replacement
template = """    
import os, timeit, csv\n
r1 = {} # original code \n
r2 = {}  # api replace \n 
if isinstance(r1, pd.Series):\n
    assert r1.equals(r2)\n
     
if isinstance(r1, pd.DataFrame):\n
    assert r1.equals(r2)\n   
else:\n
    assert np.allclose(r1,r2)\n
def func1():\n
    {}\n
def func2():\n
    {}\n
number_of_runs = 100 \n
t1 = timeit.timeit(func1, number=number_of_runs) / number_of_runs\n
t2 = timeit.timeit(func2, number=number_of_runs) / number_of_runs\n
with open("../optimization.csv", "w+") as f:\n
     writer = csv.writer(f)\n
     writer.writerow([{},{},{},t1, t2])\n
os._exit(0)\n
     """


class CallParser(ast.NodeVisitor):
    def __init__(self):
        self.attrs = []
        self.names = []


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
                name = node.func.attr
                self.names.append(name)
                self.attrs.append(node)
        elif isinstance(node.func, Name) and node.func.id in api_name:
                name = node.func.id
                self.names.append(name)
                self.attrs.append(node)


    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.value, Attribute) and node.value.attr in api_name:
            name = node.value.attr
            self.names.append(name)
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


class APIReplace(object):
    def __init__(self, code_path, option='static'):
        self.code_path = code_path
        self.option = option

    def recommend(self):
        cnt = 0
        index = 0
        # open file
        with open(self.code_path, 'r', encoding='UTF-8') as file:
            content = file.read()
        file.close()

        # parse
        tree = ast.parse(content)  # parse the code into ast_tree
        v = CallParser()   # find the potential api
        v.visit(tree)
        names = v.names

        for candidate in v.attrs:
                oldstmt = astor.to_source(candidate).strip()
                lineno = candidate.lineno
                conqs1 = self.replace_one(candidate,names[index])
                conqs2 = self.replace_two(candidate)
                conqs3 = self.replace_three(candidate)
                conqs4 = self.replace_four(candidate)
                if self.option == 'static':
                    if conqs1 or conqs2 or conqs3 or conqs4:
                        print("original API:" + oldstmt)
                    if conqs1:
                        print('Recommend API:' + conqs1)
                    if conqs2:
                        print('Recommend API:' + conqs2)
                    if conqs3:
                        print('Recommend API:' + conqs3)
                    if conqs4:
                        print('Recommend API:' + conqs4)
                    if conqs1 or conqs2 or conqs3 or conqs4:
                        print("lineno:{}".format(lineno))
                        print('###############################################')

                if self.option == 'dynamic':
                    if conqs1 or conqs2 or conqs3 or conqs4:
                        print("original API:" +oldstmt)
                    if conqs1:
                        newstmt = conqs1
                        print('Recommend API:' + newstmt)
                        self.add_source(oldstmt, newstmt, content, lineno, cnt)
                    if conqs2:
                        newstmt = conqs2
                        print('Recommend API:' + newstmt)
                        self.add_source(oldstmt, newstmt, content, lineno, cnt)
                    if conqs3:
                        newstmt = conqs3
                        print('Recommend API:' + newstmt)
                        self.add_source(oldstmt, newstmt, content, lineno, cnt)
                    if conqs4:
                        newstmt = conqs4
                        print('Recommend API:' + newstmt)
                        self.add_source(oldstmt, newstmt, content, lineno, cnt)
                    if conqs1 or conqs2 or conqs3 or conqs4:
                        cnt += 1
                        print("lineno:{}".format(lineno))
                        print('###############################################')
                index = index + 1

    def add_source(self, oldstmt, newstmt, content, lineno, cnt):
        to_add = template.format(oldstmt, newstmt, oldstmt, newstmt, lineno, '"{}"'.format(oldstmt),
                                 '"{}"'.format(newstmt))
        # fill in the template
        to_insert = ast.parse(to_add)
        # insert the new node
        temp_tree = ast.parse(content)
        CodeInstrumentator(lineno, to_insert).visit(temp_tree)
        instru_source = astor.to_source(temp_tree)
        des_path = os.path.join(os.path.dirname(self.code_path), 'code_{}.py'.format(cnt))
        with open(des_path, 'w') as wf:
            wf.write(instru_source)
        wf.close()
        self.execute_code(des_path)

    def execute_code(self, des_path):
        try:
            cmd = "python " + des_path
            print("executing:" + des_path)
            p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, err = p.communicate(timeout=600)
            err = str(err).split('\\r\\n')
            if p.returncode == 1:
                error_type = err[len(err) - 2]
                print("execution interrupt:" + error_type)
            else:
                with open("../optimization.csv", "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        t1 = row[3]
                        t2 = row[4]
                        print('origin time:{} , recommend time:{}'.format(t1, t2))
                        break
                f.close()


        except Exception as e:
                print(e)

    def replace_one(self, node, name):
        oldstmt = astor.to_source(node).strip()
        if name in api_pair.keys():
            newstmt = oldstmt.replace(name,api_pair[name])
            return  newstmt

    def replace_two(self, node):
        oldstmt = astor.to_source(node).strip()
        # np.column_stack -> np.vstack
        if '.column_stack' in oldstmt:
            string = re.match('np.column_stack\([\(\[](.*)[\)\]]\)$', oldstmt).group(1)
            args = string.split(',')
            args = map(lambda arg: '%s%s' % (arg, '.T'), args)
            args = ','.join(list(args))
            newstmt = 'np.vstack((' + args + ')).T'
            return newstmt
            #   pd.Series.str ->pd.Series.map
        elif '.str' in oldstmt:
            newstmt = oldstmt.replace('str', 'map(lambda x: x') + ')'
            return newstmt

        elif '.arrange' in oldstmt:
            newstmt = oldstmt.replace('arrange(', 'np.array(range(')
            return newstmt

        # np.ones -> np.empty, np.ndarray.fill
        elif '.ones' in oldstmt:
            newstmt = oldstmt.replace("np.ones", "np.empty") + "; r2.fill(1)"
            return newstmt
        # np.zeros -> np.empty
        elif '.zeros' in oldstmt:
            newstmt = oldstmt.replace("zeros", "empty") + "; r2[:]= 0"
            return newstmt

        # np.c_ -> np.hstack
        elif '.c_' in oldstmt:
            newstmt = oldstmt.replace("c_", "hstack(") + ")"
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

    def replace_three(self, node):
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

    def replace_four(self, node):
        oldstmt = astor.to_source(node).strip()
        # np.where->pd.DataFrame.apply
        if re.match('np.where\(.*\,.*\,.*\)$', oldstmt):
            for c in ast.iter_child_nodes(node):
                if isinstance(c, ast.Compare):
                    o = astor.to_source(c.left).strip()
                    condition = re.match('np.where\((.*)\,(.*)\,(.*)\)$', oldstmt).group(1)
                    t = re.match('np.where\((.*)\,(.*)\,(.*)\)$', oldstmt).group(2)
                    f = re.match('np.where\((.*)\,(.*)\,(.*)\)$', oldstmt).group(3)
                    newstmt = 'pd.DataFrame(' + o + ').apply(lambda x : ' + condition.replace(o,
                                                                                              'x') + ').replace({True:' + t + ',False:' + f + '}).values.flatten()'
                    return newstmt

        # np.where->pd.Series.map
        elif re.match('np.where\(.*\,.*\,.*\)$', oldstmt):
            for c in ast.iter_child_nodes(node):
                if isinstance(c, ast.Compare):
                    o = astor.to_source(c.left).strip()
                    condition = re.match('np.where\((.*)\,(.*)\,(.*)\)$', oldstmt).group(1)
                    t = re.match('np.where\((.*)\,(.*)\,(.*)\)$', oldstmt).group(2)
                    f = re.match('np.where\((.*)\,(.*)\,(.*)\)$', oldstmt).group(3)
                    newstmt = '(' + o + ').map(lambda x :' + t + ' if ' + condition.replace(o,
                                                                                            'x') + ' else ' + f + ').values'
                    return newstmt

        # np.hstack -> np.concatenate
        elif re.match('np.hstack.*', oldstmt):
            for node in ast.iter_child_nodes(node):
                if not isinstance(node, ast.Attribute):
                    agr = astor.to_source(node).strip()
                    newstmt = 'np.concatenate(' + agr + ', axis=1)'
                    return newstmt

        # np.vstack -> np.concatenate
        elif re.match('np.vstack.*', oldstmt):
            for node in ast.iter_child_nodes(node):
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
                return newstmt
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
                return newstmt
            elif ('axis=-2' in agrument2) or ('axis=0' in agrument2):
                newstmt = "np.einsum('ij->j'," + agrument1 + ")"
                return newstmt
            else:
                newstmt = "np.einsum('i->'," + agrument1 + ")"
                return newstmt




        # np.hstack->np.append
        elif re.match('np.hstack.*', oldstmt):
            for node in ast.iter_child_nodes(node):
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
            newstmt = z.replace(x, o) + '.apply(lambda ' + x + ':' + t.replace(z, x) + 'if ' + condition.replace(z,
                                                                                                                 x) + 'else ' + f.replace(
                z, x) + ')'
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













