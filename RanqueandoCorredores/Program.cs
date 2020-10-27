using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using System;
using System.Linq;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace RanqueandoCorredores
{
    class Program
    {
        /*
         *  Código para classificar a habilidade de corredores de acordo com o histórico de corridas deles 
         *  usando distribuição Gaussiana através da biblioteca Infer.Net.
         *  
         */
        static void Main(string[] args)
        {
            var nomesDosCorredores = new string[] { 
                "Corredor_zero",
                "Corredor_um",
                "Corredor_dois",
                "Corredor_três",
                "Corredor_quatro",
            };

            var IDsDosPrimeirosColocados = new int[] { 0, 3, 0, 0, 1, 2 };
            var IDsDosSegundosColocados  = new int[] { 1, 2, 2, 4, 0, 0 };
            var IDsDosTerceirosColocados = new int[] { 2, 1, 1, 1, 2, 1 };
            var IDsDosQuartosColocados   = new int[] { 3, 0, 4, 2, 4, 4 };
            var IDsDosQuintosColocados   = new int[] { 4, 4, 3, 3, 3, 3 };

            var quantidadeDosCorredores = new Range(nomesDosCorredores.Length);
            var quantidadeDeCorridas = new Range(IDsDosPrimeirosColocados.Length);

            var habilidades = Variable.Array<double>(quantidadeDosCorredores);
            habilidades[quantidadeDosCorredores] = Variable.GaussianFromMeanAndVariance(6, 9).ForEach(quantidadeDosCorredores);

            var primeiros = Variable.Array<int>(quantidadeDeCorridas);
            var segundos = Variable.Array<int>(quantidadeDeCorridas);
            var terceiros = Variable.Array<int>(quantidadeDeCorridas);
            var quartos = Variable.Array<int>(quantidadeDeCorridas);
            var quintos = Variable.Array<int>(quantidadeDeCorridas);

            primeiros.ObservedValue = IDsDosPrimeirosColocados;
            segundos.ObservedValue = IDsDosSegundosColocados;
            terceiros.ObservedValue = IDsDosTerceirosColocados;
            quartos.ObservedValue = IDsDosQuartosColocados;
            quintos.ObservedValue = IDsDosQuintosColocados;

            using (Variable.ForEach(quantidadeDeCorridas))
            {
                var performanceDoPrimeiro = Variable.GaussianFromMeanAndVariance(habilidades[primeiros[quantidadeDeCorridas]], 1);
                var performanceDoSegundo = Variable.GaussianFromMeanAndVariance(habilidades[segundos[quantidadeDeCorridas]], 1);
                var performanceDoTerceiro = Variable.GaussianFromMeanAndVariance(habilidades[terceiros[quantidadeDeCorridas]], 1);
                var performanceDoQuarto = Variable.GaussianFromMeanAndVariance(habilidades[quartos[quantidadeDeCorridas]], 1);
                var performanceDoQuinto = Variable.GaussianFromMeanAndVariance(habilidades[quintos[quantidadeDeCorridas]], 1);

                Variable.ConstrainTrue(performanceDoPrimeiro > performanceDoSegundo);
                Variable.ConstrainTrue(performanceDoSegundo > performanceDoTerceiro);
                Variable.ConstrainTrue(performanceDoTerceiro > performanceDoQuarto);
                Variable.ConstrainTrue(performanceDoQuarto > performanceDoQuinto);
            }

            var mecanismoDeInferencia = new InferenceEngine
            {
                Algorithm = new ExpectationPropagation(),
                NumberOfIterations = 50,
                ShowProgress = false,
            };

            var habilidadesInferidas = mecanismoDeInferencia.Infer<Gaussian[]>(habilidades);

            var corredores = habilidadesInferidas
                .Select((h, i) => new { Nome = nomesDosCorredores[i], Habilidade = h.GetMean() })
                .OrderByDescending(e => e.Habilidade);

            foreach (var corredor in corredores)
            {
                Console.WriteLine($"{corredor.Nome}: {corredor.Habilidade:F2}");
            }

        }
    }
}
