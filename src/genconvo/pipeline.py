"""
    Verdict pipeline for the GenConvoBench dataset. https://arxiv.org/pdf/2506.06266

    VerdictPipeline(
        context=Context(
            name="financebench",
            path="data/financebench/2022_amd_10k.txt"
        ),
        prompts=Prompts(
            question_generation=QuestionGenerationPrompts(),
            answer_generation=AnswerGenerationPrompts()
        )
    ) -->

    Pipeline(
        Unit(QuestionGeneration, StructuredOutput(16x Sonnet-3.7)),
        Unit(AnswerGeneration, StructuredOutput(16x Sonnet-3.7)),
        MapUnit(WriteToWandb)
    ) \forall question prompts in prompts.templates
"""