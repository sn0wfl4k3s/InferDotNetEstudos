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
         */
        static void Main(string[] args)
        {
            var nomesDosEquipes = new string[] { "Angels", "Bruins", "Comets", "Demons", "Eagles", "Flyers" };

            var IDsDasEquipesVencedoras = new int[] { 0, 2, 1, 0, 1, 3, 0, 2, 4 };
            var IDsDasEquipesPerdedoras = new int[] { 1, 3, 2, 4, 3, 5, 5, 4, 5 };

            var quantidadeDeEquipes = new Range(nomesDosEquipes.Length);
            var quantidadeDePartidas = new Range(IDsDasEquipesVencedoras.Length);


            var habilidades = Variable.Array<double>(quantidadeDeEquipes);
            habilidades[quantidadeDeEquipes] = Variable.GaussianFromMeanAndVariance(6, 9).ForEach(quantidadeDeEquipes);

            var vencedores = Variable.Array<int>(quantidadeDePartidas);
            var perdedores = Variable.Array<int>(quantidadeDePartidas);

            vencedores.ObservedValue = IDsDasEquipesVencedoras;
            perdedores.ObservedValue = IDsDasEquipesPerdedoras;

            using (Variable.ForEach(quantidadeDePartidas))
            {
                var performanceDoVencedor = Variable.GaussianFromMeanAndVariance(habilidades[vencedores[quantidadeDePartidas]], 1);
                var performanceDoPerdedor = Variable.GaussianFromMeanAndVariance(habilidades[perdedores[quantidadeDePartidas]], 1);

                Variable.ConstrainTrue(performanceDoVencedor > performanceDoPerdedor);
            }

            var mecanismoDeInferência = new InferenceEngine
            {
                Algorithm = new ExpectationPropagation(),
                NumberOfIterations = 50,
                ShowProgress = false,
            };

            var habilidadesInferidas = mecanismoDeInferência.Infer<Gaussian[]>(habilidades);

            for (int i = 0; i < nomesDosEquipes.Length; ++i)
            {
                var habilidade = habilidadesInferidas[i].GetMean();

                Console.WriteLine($"{nomesDosEquipes[i]}: {habilidade:F2}");
            }
        }
    }
}
