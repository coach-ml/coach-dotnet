using System;
using System.Threading.Tasks;
using Coach;

namespace CoachSample
{
    class Program
    {
        static void Main(string[] args)
        {
            Login().Wait();
            Console.WriteLine("Done");
        }

        static async Task Login() {
            var c = new CoachClient();
            await c.Login("");
            await c.CacheModel("flowers");
            
            var model = c.GetModel("flowers");
            var eval = model.Predict("rose.jpg");

            Console.WriteLine($"{eval.Label} : {eval.Confidence}");
        }
    }
}
