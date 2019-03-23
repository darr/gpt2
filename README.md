# gpt2

#### Description
implement OpenAI gpt2

## How To Run?

```shell
bash run.sh
```

Yes, just run the shell, it can work.
run.sh will create a virtual env that needed by gpt, and install all  
software that needed by gpt2.  

And how about the pretrained model ,config and dataset that we need  
for finetune?  

All of them will be download and cached after we run the shell run.sh  

In a word, just run the shell:  
```shell
bash run.sh
```
Then you get a result.  

And if you need,read the code.  

## papers

[Gaussian Error Linear Units](https://arxiv.org/pdf/1606.08415.pdf)  
[translate to chinese](./papers/GAUSSIAN_ERROR_LINEAR_UNITS1606.08415.md)  

[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)  
[translate to chinese](./papers/Attention_Is_All_You_Need1706.03762.md)  

[Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)  
[translate to chinese](./papers/Improving_Language_Understanding_by_Generative_Pre-Training.md)  

[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)  
[translate to chinese](./papers/Language_Models_are_Unsupervised_Multitask_Learners.md)  


## write story

## input text 

```shell
I love my motherland.
```

## output

```shell
She saved my life."

The statement ends with the following line: "When I say I'm sorry, your daughter was killed."<|endoftext|>"I'd like to give you an insight into my family history," Trump said of the elder Johnson, saying that he was called into their home twice during a visit to Washington in the summer of 1996. "For some reason it was just in the background and that was it, but during that time I became very focused on the world and I decided, like a lot of people before me, that I was doing a lot of things that I never dreamed I would. These is a young woman that I started off being very conscious of but also knew where she belonged." "And then things became really serious over the summer," Trump continued. "And I started to pick things up and I've always felt there was a certain little thing about them that allowed me, in my life, to get really serious about that — or to feel really deep about those things and to really start, like, making connections if you will with what my family is trying to do, even for people and their kids and my brother, a lot of people were like, 'I don't know how, you do'. But just like there was a small corner of my life where I felt I had a certain way of going about everything."<|endoftext|>The United States government is using a law to force anyone to sign up for immigration legal aid who doesn't fit a comprehensive health insurance plan. The law has been passed in eight states last year to allow folks to skip insurance for themselves and their dependent children. The first step in establishing legal identification takes place a couple of weeks prior to next month's vote, and likely could pass, Sen. Patty Murray told reporters Tuesday at a Senate hearing on immigration reform.

Advertisement

If the law does pass, it would essentially extend the stay so you don't have to pay for essential health care, including those for your children. As many people point out, Medicaid in some other states is already in place to cover nonpregnant people, so not as many Americans may be getting adequate insurance coverage through that program as required by Federal law. The only thing a states' version of the law will do is tell people who might come to the country legally that the law's health care requirements apply.

I will probably be able to find a lawyer as quickly as possible to bring this issue to the attention of the House, and the Senate, and I will take the opportunity to do
```


## Dataset

use the dataset ROCStories

## run directly

```shell
show.py line:42 ***** Eval results *****
show.py line:44 eval_accuracy = 0.5531801175841796
show.py line:44 eval_loss = 0.6910396269244007
show.py line:44 train_loss = 0.0
```

## run finetuned


```shell
show.py line:42 ***** Eval results *****
show.py line:44 eval_accuracy = 0.8273650454302512
show.py line:44 eval_loss = 0.36625397409129345
show.py line:44 train_loss = 3.138157744692941
```

The result is not so well, I doubt there is something wrong with it.
The gpt2 result eval accuracy is less than gpt result.
I did not use special tokens in gpt2.
here is the gpt result

```shell
show.py line:42 ***** Eval results *****
show.py line:44 eval_accuracy = 0.863174772848744
show.py line:44 eval_loss = 0.31887995107815814
show.py line:44 train_loss = 3.087455103540013
```

with the epochs increasing, the eval accuracy of gpt is still better than gpt2.
