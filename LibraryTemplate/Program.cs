using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using static System.Math;
using static AtCoderTemplate.MyConstants;
using static AtCoderTemplate.MyInputOutputs;
using static AtCoderTemplate.MyNumericFunctions;
using static AtCoderTemplate.MyAlgorithm;
using static AtCoderTemplate.MyDataStructure;
using static AtCoderTemplate.MyExtensions;
using static AtCoderTemplate.MyEnumerable;

namespace AtCoderTemplate
{
    public class Program
    {
        public static void Main(string[] args) { }
    }

    public static class MyInputOutputs
    {
        /* Input & Output*/
        public static string Read()
        {
            return Console.ReadLine();
        }

        public static List<string> Reads()
        {
            return Console.ReadLine().Split(' ').ToList();
        }

        public static List<List<string>> ReadRows(int rowNum)
        {
            /*
            入力例
            A1 B1 C1 ... Z1
            A2 B2 C2 ... Z2
            ...
            An Bn Cn ... Zn
           

            出力例
            [[A1, B1, C1, ... Z1], [A2, B2, C2, ... Z2], ... [An, Bn, Cn, ... Zn]]
            */
            return Enumerable.Range(0, rowNum).Select(i => Reads()).ToList();
        }

        public static List<List<string>> ReadColumns(int rowNum, int colNum)
        {
            /*
            入力例
            A1 B1 C1 ... Z1
            A2 B2 C2 ... Z2
            ...
            An Bn Cn ... Zn
           

            出力例
            [[A1, A2, A3, ... An], [B1, B2, B3, ... Bn], ... [Z1, Z2, Z3, ... Zn]]
            */
            var rows = ReadRows(rowNum);
            return Enumerable.Range(0, colNum).Select(i => rows.Select(items => items[i].ToString()).ToList()).ToList();
        }

        public static List<List<string>> ReadGridGraph(int height, int width)
        {
            /*
            入力例
            A1B1C1...Z1
            A2B2C2...Z2
            ...
            AnBnCn...Zn
           

            出力例
            [[A1, B1, C1, ... Z1], [A2, B2, C2, ... Z2], ... [An, Bn, Cn, ... Zn]]
            */
            return Enumerable.Range(0, height)
                .Select(i =>
                   Read()
                   .Select(c => c.ToString())
                   .ToList()
                ).ToList();
        }

        public static int ToInt(this string str)
        {
            return int.Parse(str);
        }

        public static long ToLong(this string str)
        {
            return long.Parse(str);
        }

        public static List<int> ToInts(this List<string> strs)
        {
            return strs.Select(str => str.ToInt()).ToList();
        }

        public static List<long> ToLongs(this List<string> strs)
        {
            return strs.Select(str => str.ToLong()).ToList();
        }

        public static int ReadInt()
        {
            return Read().ToInt();
        }
        public static long ReadLong()
        {
            return Read().ToLong();
        }

        public static List<int> ReadInts()
        {
            return Reads().ToInts();
        }
        public static List<long> ReadLongs()
        {
            return Reads().ToLongs();
        }

        public static void Print<T>(T item)
        {
            Console.WriteLine(item);
        }
        public static void PrintIf<T1, T2>(bool condition, T1 trueResult, T2 falseResult)
        {
            if (condition)
            {
                Console.WriteLine(trueResult);
            }
            else
            {
                Console.WriteLine(falseResult);
            }
        }

        public static void PrintRow<T>(IEnumerable<T> list)
        {
            /* 横ベクトルで表示
            A B C D ...
            */
            if (!list.IsEmpty())
            {
                Console.Write(list.First());
                foreach (var item in list.Skip(1))
                {
                    Console.Write($" {item}");
                }
            }
            Console.Write("\n");
        }
        public static void PrintColomn<T>(IEnumerable<T> list)
        {
            /* 縦ベクトルで表示
            A
            B
            C
            D
            ...
            */
            foreach (var item in list)
            {
                Console.WriteLine(item);
            }
        }
        public static void PrintRows<T>(IEnumerable<IEnumerable<T>> sources)
        {
            foreach (var row in sources)
            {
                PrintRow(row);
            }
        }

        public static void PrintGridGraph<T>(IEnumerable<IEnumerable<T>> sources)
        {
            foreach (var row in sources)
            {
                Print(String.Concat(row));
            }
        }
    }

    public static class MyConstants
    {
        public static IEnumerable<char> lowerAlphabets = Enumerable.Range('a', 'z' - 'a' + 1).Select(i => (char)i);
        public static IEnumerable<char> upperAlphabets = Enumerable.Range('A', 'Z' - 'A' + 1).Select(i => (char)i);

        public static int p1000000007 = (int)Pow(10, 9) + 7;
    }

    public static class MyNumericFunctions
    {

        public static bool IsEven(int a)
        {
            return a % 2 == 0;
        }
        public static bool IsOdd(int a)
        {
            return !IsEven(a);
        }
        public static bool IsEven(long a)
        {
            return a % 2L == 0L;
        }
        public static bool IsOdd(long a)
        {
            return !IsEven(a);
        }

        public static int[,] PascalsTriangle(int nmax, int kmax, int divisor)
        {
            var comb = new int[2000 + 1, 2000 + 1];
            foreach (var n in MyEnumerable.Interval(0, 2000 + 1))
            {
                foreach (var k in MyEnumerable.Interval(0, 2000 + 1))
                {
                    if (n < k) continue;

                    if (k == 0)
                    {
                        comb[n, k] = 1;
                    }
                    else
                    {
                        comb[n, k] = (int)(((long)comb[n - 1, k - 1] + comb[n - 1, k]) % divisor);
                    }
                }
            }

            return comb;
        }
        /// <summary>
        /// Mod計算
        /// </summary>
        public class Mods
        {
            int divisor;
            public Mods(int divisor)
            {
                this.divisor = divisor;
            }
            public int Mod(long a)
            {
                var b = (int)(a % divisor);
                if (b < 0)
                {
                    return b + divisor;
                }
                else
                {
                    return b;
                }
            }

            public int Add(int a, int b)
            {
                return Mod(Mod(a) + Mod(b));
            }
            public int Sub(int a, int b)
            {
                return Mod(Mod(a) - Mod(b));
            }

            public int Mul(int a, int b)
            {
                return Mod((long)Mod(a) * Mod(b));
            }

            public int Pow(int b, int n)
            {
                var digit = (int)Math.Log(n, 2.0);
                var pows = Interval(0, digit + 1)
                    .Scanl(b, (accm, _) => Mul(accm, accm))
                    .ToArray();
                return Interval(0, digit + 1)
                    .Aggregate(1, (accm, i) => ((n >> i) & 1) == 1 ? Mul(accm, pows[i]) : accm);
            }
            public int Inv(int a)
            {
                return Pow(a, divisor - 2);
            }
            public int Div(int a, int b)
            {
                return Mul(a, Inv(b));
            }

            public int Perm(int n, int k)
            {
                if (n < 0 || k < 0) throw new ArgumentOutOfRangeException();

                if (n < k)
                {
                    return 0;
                }
                else
                {
                    return Interval(n - k + 1, n + 1)
                        .Aggregate(1, Mul);
                }
            }

            public List<int> FactTable(int nMax)
            {
                return Interval(1, nMax + 1)
                    .Scanl(1, Mul)
                    .ToList();
            }

            public List<List<int>> CombTable(int nMax)
            {
                var table = Enumerable.Repeat(0, nMax + 1)
                    .Select(_ =>
                       Enumerable.Repeat(0, nMax + 1).ToList()
                    ).ToList();

                foreach (var n in Interval(0, nMax + 1))
                {
                    foreach (var k in Interval(0, nMax + 1))
                    {
                        if (n < k)
                        {
                            table[n][k] = 0;
                        }
                        else if (k == 0)
                        {
                            table[n][k] = 1;
                        }
                        else
                        {
                            table[n][k] = Add(table[n - 1][k - 1], table[n - 1][k]);
                        }
                    }
                }
                return table;
            }
        }

        /// <summary>
        /// 最大公約数を得る 
        /// O(log N)
        /// </summary>
        /// <param name="m">自然数</param>
        /// <param name="n">自然数</param>
        /// <returns></returns>
        public static long GCD(long m, long n)
        {
            // GCD(m,n) = GCD(n, m%n)を利用
            // m%n = 0のとき、mはnで割り切れるので、nが最大公約数
            if (m <= 0L || n <= 0L) throw new ArgumentOutOfRangeException();

            if (m < n) return GCD(n, m);
            while (m % n != 0L)
            {
                var n2 = m % n;
                m = n;
                n = n2;
            }
            return n;
        }

        /// <summary>
        /// 最小公倍数を得る
        /// O(log N)
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        public static long LCM(long m, long n)
        {
            var ans = checked((long)(BigInteger.Multiply(m, n) / GCD(m, n)));
            return ans;
        }

        /// <summary>
        /// 約数列挙(非順序)
        /// O(√N)
        /// </summary>
        /// <param name="m">m > 0</param>
        /// <returns></returns>
        public static IEnumerable<long> Divisor(long m)
        {
            if (!(m > 0)) throw new ArgumentOutOfRangeException();

            var front = Enumerable.Range(1, (int)Sqrt(m))
                .Select(i => (long)i)
                .Where(d => m % d == 0);
            return front.Concat(front.Where(x => x * x != m).Select(x => m / x));
        }

        public static IEnumerable<int> Divisor(int m)
        {
            if (!(m > 0)) throw new ArgumentOutOfRangeException();

            var front = Enumerable.Range(1, (int)Sqrt(m))
                .Where(d => m % d == 0);
            return front.Concat(front.Where(x => x * x != m).Select(x => m / x));
        }

        /// <summary>
        /// 公約数列挙(非順序)
        /// O(√N)
        /// </summary>
        /// <param name="m">m > 0</param>
        /// <param name="n">n > 0 </param>
        /// <returns></returns>
        public static IEnumerable<long> CommonDivisor(long m, long n)
        {
            if (!(m > 0)) throw new ArgumentOutOfRangeException();

            if (m < n) return CommonDivisor(n, m);
            return Divisor(m).Where(md => n % md == 0);
        }

        public static IEnumerable<int> CommonDivisor(int m, int n)
        {
            if (!(m > 0)) throw new ArgumentOutOfRangeException();

            if (m < n) return CommonDivisor(n, m);
            return Divisor(m).Where(md => n % md == 0);
        }

        /// <summary>
        /// エラトステネスの篩 O(N loglog N)
        /// </summary>
        /// <param name="n"></param>
        /// <returns></returns>
        public static IEnumerable<int> Primes(int n)
        {
            if (!(n > 1)) throw new ArgumentOutOfRangeException();

            var ps = Interval(2, n + 1);
            while (!ps.IsEmpty() && ps.First() <= Sqrt(n))
            {
                var m = ps.First();
                ps = ps.Where(p => p % m != 0);
            }
            return ps;
        }

        /// <summary>
        /// 素因数分解 O(N loglog N)
        /// </summary>
        /// <param name="n"></param>
        /// <returns></returns>
        public static IEnumerable<int> PrimeFactrization(int n)
        {
            if (!(n > 1)) throw new ArgumentOutOfRangeException();

            var e = new int[n + 1];
            var p = n;
            var ps = Primes(n).ToList();
            var i = 0;
            while (p != 1)
            {
                if (p % ps[i] == 0)
                {
                    e[ps[i]] += 1;
                    p /= ps[i];
                    continue;
                }
                i += 1;
            }
            return e;
        }

        /// <summary>
        /// 順列を得る
        /// O(N!)
        /// </summary>
        public static IEnumerable<IEnumerable<T>> Perms<T>(IEnumerable<T> source, int n)
        {
            if (n == 0 || source.IsEmpty() || source.Count() < n)
            {
                return Enumerable.Empty<IEnumerable<T>>();
            }
            else if (n == 1)
            {
                return source.Select(i => new List<T> { i });
            }
            else
            {
                var nexts = source.Select((x, i) =>
                   new { next = source.Take(i).Concat(source.Skip(i + 1)), selected = source.Take(i + 1).Last() });
                return nexts.SelectMany(next => Perms(next.next, n - 1).Select(item => item.Prepend(next.selected)));
            }
        }
    }

    public static class MyAlgorithm
    {
        /// <summary>
        /// めぐる式二分探索法
        /// O(log N)
        /// </summary>
        /// <param name="list">探索するリスト</param>
        /// <param name="predicate">条件の述語関数</param>
        /// <param name="ng">条件を満たさない既知のindex</param>
        /// <param name="ok">条件を満たす既知のindex</param>
        /// <returns>条件を満たすindexの内、隣がfalseとなるtrueのindexを返す</returns>
        public static long BinarySearch<T>(Func<long, bool> predicate, long ng, long ok)
        {
            while (Abs(ok - ng) > 1)
            {
                var mid = (ok + ng) / 2L;
                if (predicate(mid))
                {
                    ok = mid;
                }
                else
                {
                    ng = mid;
                }
            }
            return ok;
        }

        /// <summary>
        /// ランレングス符号化
        /// </summary>
        /// <param name="source">元の文字列</param>
        /// <returns>連続した文字をひとまとめにし、その文字と長さのペアの列を得る</returns>
        /// <example>RunLengthEncoding("aaabccdddd") => [(a,3), (b,1), (c,2), (d,4)]</example>
        public static IEnumerable<Tuple<string, int>> RunLengthEncoding(string source)
        {
            var cutIndexes = Interval(1, source.Length)
                .Where(i => source[i] != source[i - 1])
                .Prepend(0)
                .Append(source.Length);
            return cutIndexes
                .MapAdjacent((i0, i1) => Tuple.Create<string, int>(source[i0].ToString(), i1 - i0));
        }

        /// <summary>
        /// 3分探索法 O(log N)
        /// </summary>
        /// <param name="l">定義域の最小値</param>
        /// <param name="r">定義域の最大値</param>
        /// <param name="f">凸な関数</param>
        /// <param name="allowableError">許容誤差</param>
        /// <param name="isDownwardConvex">下に凸か（falseならば上に凸）</param>
        /// <returns>凸関数f(x)の許容誤差を含む極値への元を得る</returns>
        public static double TernarySerch(double l, double r, Func<double, double> f, double allowableError, bool isDownwardConvex)
        {
            while (r - l >= allowableError)
            {
                var ml = l + (r - l) / 3; // mid left
                var mr = l + (r - l) / 3 * 2.0; // mid right
                var fml = f(ml);
                var fmr = f(mr);
                if (isDownwardConvex)
                {
                    if (fml < fmr)
                    {
                        r = mr;
                    }
                    else if (fml > fmr)
                    {
                        l = ml;
                    }
                    else
                    {
                        l = ml;
                        r = mr;
                    }
                }
                else
                {
                    if (fml < fmr)
                    {
                        l = ml;
                    }
                    else if (fml > fmr)
                    {
                        r = mr;
                    }
                    else
                    {
                        l = ml;
                        r = mr;
                    }
                }
            }
            return l;
        }

        /// <summary>
        /// フラグが立っているか判定する
        /// </summary>
        /// <param name="flags">フラグ</param>
        /// <param name="digit">判定する桁目</param>
        /// <returns>digit のフラグが立っていれば true</returns>
        public static bool IsFlag(long flags, int digit)
        {
            return ((flags >> digit) & 1L) == 1L;
        }
    }

    public static class MyDataStructure
    {
        public class UnionFind
        {
            List<int> parent;
            List<int> size;
            public UnionFind(int N)
            {
                // 最初はすべて異なるグループ番号(root)が割り当てられる
                parent = Enumerable.Range(0, N).ToList();
                size = Enumerable.Repeat(1, N).ToList();
            }

            // 頂点uの属するグループ番号(root)を探す
            int Root(int u)
            {
                if (parent[u] == u)
                {
                    return u;
                }
                else
                {
                    var root = Root(parent[u]);
                    parent[u] = root; // 経路圧縮
                    return root;
                }
            }

            // 2つのグループを統合する(rootが異なる場合、同じrootにする)
            public void Unite(int u, int v)
            {
                int root_u = Root(u);
                int root_v = Root(v);
                if (root_u == root_v) return;
                parent[root_u] = root_v; // root_vをroot_uの親とする
                size[root_v] += size[root_u];
            }

            public int Size(int u)
            {
                return size[Root(u)];
            }

            public bool IsSame(int u, int v)
            {
                int root_u = Root(u);
                int root_v = Root(v);
                return root_u == root_v;
            }
        }

        /// <summary>
        /// Comparison<T>(順序を定める高階関数 t -> t -> Orderingみたいなもの)からIComparer<T>の実体へ変換するクラス
        /// </summary>
        /// <typeparam name="T">順序付けられる型</typeparam>
        public class ComparisonToComparerConverter<T> : IComparer<T>
        {
            Comparison<T> comparison;
            public ComparisonToComparerConverter(Comparison<T> comparison)
            {
                this.comparison = comparison;
            }
            public int Compare(T x, T y)
            {
                return comparison(x, y);
            }
        }

        /// <summary>
        /// IComparble<T>を持つ型Tから、その逆順序であるComparison<T>を得る
        /// </summary>
        /// <typeparam name="T">順序付きの型</typeparam>
        /// <returns>IComparable<T>の逆順序</returns>
        public static Comparison<T> ReverseOrder<T>() where T : IComparable<T>
        {
            return (x, y) => -x.CompareTo(y);
        }

        /// <summary>
        /// 優先度付きキュー(デフォルトでは昇順)
        /// 先頭参照・要素数がO(1)、要素の追加・先頭削除がO(log N)
        /// </summary>
        /// <typeparam name="T">順序付きの型</typeparam>
        public class PriorityQueue<T> : IEnumerable<IEnumerable<T>>
            where T : IComparable<T>
        {
            SortedDictionary<T, Queue<T>> dict;
            int size = 0;

            public PriorityQueue()
            {
                dict = new SortedDictionary<T, Queue<T>>();
            }

            public PriorityQueue(IComparer<T> comparer)
            {
                dict = new SortedDictionary<T, Queue<T>>(comparer);
            }

            public PriorityQueue(Comparison<T> comparison)
            {
                dict = new SortedDictionary<T, Queue<T>>(new ComparisonToComparerConverter<T>(comparison));
            }

            public void Enqueue(T item)
            {
                if (dict.ContainsKey(item))
                {
                    dict[item].Enqueue(item);
                }
                else
                {
                    var added = new Queue<T>();
                    added.Enqueue(item);
                    dict.Add(item, added);
                }
                size += 1;
            }

            public T Peek()
            {
                return dict.First().Value.First();
            }

            public int Size()
            {
                return size;
            }

            public T Dequeue()
            {
                var first = dict.First();
                if (first.Value.Count <= 1)
                {
                    dict.Remove(first.Key);
                }
                size -= 1;
                return first.Value.Dequeue();
            }

            public IEnumerator<IEnumerable<T>> GetEnumerator()
            {
                foreach (var kv in dict)
                {
                    yield return kv.Value;
                }
            }

            IEnumerator IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }
    }

    public static class MyExtensions
    {
        public static bool IsEmpty<T>(this IEnumerable<T> source)
        {
            return !source.Any();
        }

        /// <summary>
        /// シーケンスの隣り合う要素を2引数の関数に適用したシーケンスを得る
        /// </summary>
        /// <para>O(N)</para>
        /// <param name="source">元のシーケンス</param>
        /// <param name="func">2引数関数</param>
        /// <example>[1,2,3,4].MapAdjacent(f) => [f(1,2), f(2,3), f(3,4)]</example>
        public static IEnumerable<TR> MapAdjacent<T1, TR>(this IEnumerable<T1> source, Func<T1, T1, TR> func)
        {
            var list = source.ToList();
            return Enumerable.Range(1, list.Count - 1)
                .Select(i => func(list[i - 1], list[i]));
        }

        /// <summary>
        /// 累積項を要素にもつシーケンスを得る(初項は、first)
        /// <para>O(N)</para>
        /// </summary>
        /// <param name="source">元のシーケンス</param>
        /// <param name="func">2引数関数f</param>
        /// <param name="first">func(first, source[0])のための初項</param>
        /// <example> [1,2,3].Scanl1(0,f) => [0, f(0,1), f(f(0,1),2), f(f(f(0,1),2),3)]</example>
        public static IEnumerable<TR> Scanl<T, TR>(this IEnumerable<T> source, TR first, Func<TR, T, TR> func)
        {
            var list = source.ToList();
            var result = new List<TR> { first };
            foreach (var i in Enumerable.Range(0, source.Count()))
            {
                result.Add(func(result[i], list[i]));
            }
            return result;
        }
        /// <summary>
        /// 累積項を要素にもつシーケンスを得る（初項は、source.First()）
        /// <para>O(N)</para>
        /// </summary>
        /// <param name="source">要素数1以上のシーケンス</param>
        /// <param name="func">2引数関数f</param>
        /// <example> [1,2,3].Scanl1(f) => [1, f(1,2), f(f(1,2),3)]</example>
        public static IEnumerable<T> Scanl1<T>(this IEnumerable<T> source, Func<T, T, T> func)
        {
            if (source.IsEmpty()) throw new ArgumentOutOfRangeException();
            var list = source.ToList();
            var result = new List<T> { list[0] };
            foreach (var i in Enumerable.Range(1, source.Count() - 1))
            {
                result.Add(func(result[i - 1], list[i]));
            }
            return result;
        }

        /// <summary>
        /// 数列の和をdivisorで割った余りを計算する
        /// </summary>
        /// <param name="source">数列</param>
        /// <param name="divisor">割る数</param>
        /// <returns>数列の和をdivisorで割った余り</returns>
        public static int Sum(this IEnumerable<int> source, int divisor)
        {
            return source.Aggregate(0, (a, b) => (int)(((long)a + b) % divisor));
        }

        /// <summary>
        /// 数列の積を計算する
        /// </summary>
        /// <param name="source">数列</param>
        /// <returns>数列の積</returns>
        public static long Product(this IEnumerable<long> source)
        {
            return source.Aggregate(1L, (a, b) => a * b);
        }

        /// <summary>
        /// 数列の積をdivisorで割った余りを計算する
        /// </summary>
        /// <param name="source">数列</param>
        /// <param name="divisor">割る数</param>
        /// <returns>数列の積をdivisorで割った余り</returns>
        public static int Product(this IEnumerable<int> source, int divisor)
        {
            return source.Aggregate(1, (a, b) => (int)(((long)a * b) % divisor));
        }

    }

    public static class MyEnumerable
    {
        /// <summary>
        /// 左閉右開区間 [startIndex,endIndex) を得る
        /// </summary>
        /// <param name="startIndex">始まりのインデックス。含む</param>
        /// <param name="endIndex">終わりのインデックス。含まない</param>
        public static IEnumerable<int> Interval(int startIndex, int endIndex)
        {
            if (endIndex - startIndex < 0) new ArgumentException();
            return Enumerable.Range(startIndex, endIndex - startIndex);
        }

        /// <summary>
        /// フラグから分割するindexの位置の列への変換
        /// </summary>
        /// <param name="flags">二進数によるフラグ</param>
        /// <param name="flagSize">フラグの数</param>
        /// <returns>分割するindexの位置の列</returns>
        /// <example> CutFlagToCutIndex(10110) => [0, 2, 3, 5, 6]</example>
        public static IEnumerable<int> CutFlagToIndexes(int flags)
        {
            int flagSize = (int)Log(flags, 2);
            var indexes = new List<int> { 0 };
            foreach (var i in MyEnumerable.Interval(0, flagSize))
            {
                if ((flags >> i) % 2 == 1)
                {
                    indexes.Add(i + 1);
                }
            }
            indexes.Add(flagSize + 1);
            return indexes;
        }
    }
    public static class Template
    {
        public static void TreeBFS()
        {
            // 与えられた頂点数と辺
            var N = ReadInt();
            var ab = ReadColumns(N, 2);
            var a = ab[0].ToInts();
            var b = ab[1].ToInts();

            // 木構造の初期化
            var tree = MyEnumerable.Interval(0, N)
            .Select(i => new List<int>())
            .ToList();

            foreach (var i in MyEnumerable.Interval(0, N - 1))
            {
                tree[a[i] - 1].Add(b[i] - 1);
                tree[b[i] - 1].Add(a[i] - 1);
            }

            var order = new Queue<int>();   // BFS
            // var order = new Stack<int>();   // DFS

            var reached = Enumerable.Repeat(false, N)
            .ToArray();

            // 最初の頂点を追加
            order.Enqueue(0);
            reached[0] = true;

            // 求めたい値
            var ans = false;

            // 実行
            while (!order.IsEmpty())
            {
                var node = order.Dequeue(); // 現在見ている頂点
                Func<int, bool> pruningCondition = child => false; // 枝刈り条件
                var nexts = tree[node]
                .Where(child => !reached[child] && !pruningCondition(child));

                // ************** 処理 *******************

                // 現在の頂点による処理

                // 現在の頂点と、次の頂点による処理
                foreach (var next in nexts)
                {

                }

                // ***************************************

                foreach (var next in nexts)
                {
                    order.Enqueue(next);
                    reached[next] = true;
                }
            }
        }

        public static void GridGraphBFS()
        {
            // 与えられた高さと幅のグリッドグラフ
            var HW = ReadInts();
            var H = HW[0];
            var W = HW[1];
            var S = ReadGridGraph(H, W);

            var order = new Queue<(int, int)>();
            var reached = new bool[H, W];
            foreach (var h in MyEnumerable.Interval(0, H))
            {
                foreach (var w in MyEnumerable.Interval(0, W))
                {
                    reached[h, w] = false;
                }
            }

            // 最初の頂点を追加
            order.Enqueue((0, 0));
            reached[0, 0] = true;

            // 求めたい値
            var ans = false;

            // 実行
            while (!order.IsEmpty())
            {
                var (y0, x0) = order.Dequeue();

                // 周りのマス
                var allSides = new[]{
                    (y0, x0 -1 ),   // left
                    (y0 - 1, x0),   // up
                    (y0, x0 + 1),   // right
                    (y0 + 1, x0),   // down
                }
                .Where(t =>
                {
                    var (y, x) = t;
                    return 0 <= y && y < H && 0 <= x && x < W;
                });

                // 枝刈り条件
                Func<int, int, bool> pruningCondition = (y, x) => false;

                var nexts = allSides
                    .Where(t =>
                    {
                        var (y, x) = t;
                        return !reached[y, x] && !pruningCondition(y, x);
                    });

                // **************** 処理 ******************

                // 現在の頂点による処理

                // 現在の頂点と、次の頂点による処理
                foreach (var (y, x) in nexts)
                {

                }

                // ****************************************

                foreach (var next in nexts)
                {
                    var (y, x) = next;
                    order.Enqueue(next);
                    reached[y, x] = true;
                }
            }
        }

        public static void WarshallFloyd()
        {
            // グラフの入力
            var NM = ReadInts();
            var N = NM[0];
            var M = NM[1];
            var abc = ReadColumns(M, 3);
            var a = abc[0].ToInts();
            var b = abc[1].ToInts();
            var c = abc[2].ToInts();

            // 隣接行列の初期化
            var inf = (long)Pow(2L, 60);
            var G = new long[N, N];
            foreach (var h in MyEnumerable.Interval(0, N))
            {
                foreach (var w in MyEnumerable.Interval(0, N))
                {
                    if (h == w)
                    {
                        G[h, w] = 0L;
                    }
                    else
                    {
                        G[h, w] = inf;
                    }
                }
            }
            foreach (var i in MyEnumerable.Interval(0, M))
            {
                G[a[i], b[i]] = c[i];
                G[b[i], a[i]] = c[i];
            }

            // 実行
            foreach (var k in MyEnumerable.Interval(0, N))
            {
                foreach (var i in MyEnumerable.Interval(0, N))
                {
                    foreach (var j in MyEnumerable.Interval(0, N))
                    {
                        G[i, j] = Min(G[i, j], G[i, k] + G[k, j]);
                    }
                }
            }
        }
    }
}