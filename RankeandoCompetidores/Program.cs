using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using System;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace RankeandoCompetidores
{
    class Program
    {
        /*
         *  Artigos de referência: 
         *  https://docs.microsoft.com/pt-br/dotnet/machine-learning/how-to-guides/matchup-app-infer-net
         *  https://docs.microsoft.com/pt-br/archive/msdn-magazine/2019/february/test-run-rating-competitors-using-infer-net
         *  
         */
        static void Main(string[] args)
        {
            var nomesDosTimes = new string[] { "Angels", "Bruins", "Comets", "Demons", "Eagles", "Flyers" };

            var idsDosTimesVencedores = new int[] { 0, 2, 1, 0, 1, 3, 0, 2, 4 };
            var idsDosTimesPerdedores = new int[] { 1, 3, 2, 4, 3, 5, 5, 4, 5 };

            var rangeDaQuantidadeDeTimes = new Range(nomesDosTimes.Length);
            var rangeDaquantidadeDeJogos = new Range(idsDosTimesVencedores.Length);


            var habilidades = Variable.Array<double>(rangeDaQuantidadeDeTimes);
            habilidades[rangeDaQuantidadeDeTimes] = Variable.GaussianFromMeanAndVariance(6, 9).ForEach(rangeDaQuantidadeDeTimes);

            var vencedores = Variable.Array<int>(rangeDaquantidadeDeJogos);
            var perdedores = Variable.Array<int>(rangeDaquantidadeDeJogos);

            vencedores.ObservedValue = idsDosTimesVencedores;
            perdedores.ObservedValue = idsDosTimesPerdedores;

            using (Variable.ForEach(rangeDaquantidadeDeJogos))
            {
                var performanceDoVencedor = Variable.GaussianFromMeanAndVariance(habilidades[vencedores[rangeDaquantidadeDeJogos]], 1);
                var performanceDoPerdedor = Variable.GaussianFromMeanAndVariance(habilidades[perdedores[rangeDaquantidadeDeJogos]], 1);

                Variable.ConstrainTrue(performanceDoVencedor > performanceDoPerdedor);
            }

            var mecanismoDeInferência = new InferenceEngine 
            {
                Algorithm = new ExpectationPropagation(),
                NumberOfIterations = 100,
                ShowProgress = true,
            };

            var habilidadesInferidas = mecanismoDeInferência.Infer<Gaussian[]>(habilidades);

            for (int i = 0; i < nomesDosTimes.Length; ++i)
            {
                var habilidade = habilidadesInferidas[i].GetMean();
                
                Console.WriteLine($"{nomesDosTimes[i]}: {habilidade:F2}");
            }
        }
    }
}
