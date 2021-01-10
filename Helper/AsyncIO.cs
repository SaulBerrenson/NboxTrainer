using System;
using System.Threading.Tasks;

namespace NboxTrainer.Helper
{
    internal class AsyncIO
    {
        public static Task StartTask(Action _action)
        {
          return Task.Factory.StartNew(_action);
        }
    }

  
}