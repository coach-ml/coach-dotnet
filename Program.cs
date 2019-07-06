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
            await c.Login("B1dp8OKfOk2RskZ9EkJai3rIAkENZQky2eINkWhi");
            await c.CacheModel("flowers");
            
            c.GetModel("flowers");
        }
    }
}
