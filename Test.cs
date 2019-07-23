using System;
using System.Threading.Tasks;
using Coach;

class Test
{
    static void Main(string[] args)
    {
        var model = GetStarted().Result;
        var result = model.Predict("rose.jpg").Best();

        Console.WriteLine($"{result.Label}: {result.Confidence}");
    }

    static async Task<CoachModel> GetStarted()
    {
        var c = new CoachClient();
        await c.Login("");
        await c.CacheModel("flowers", skipMatch: false);
        
        return c.GetModel("flowers");
    }
}