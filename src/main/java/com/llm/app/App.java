package com.llm.app;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        Main main = new Main();
        String prompt = "Hi, I am Linus der";
        String result = main.generateText(prompt);
        System.out.println("User: " + prompt);
        System.out.println("------------------------------");
        System.out.println("Assistant: " + result);
    }
}
