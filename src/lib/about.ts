export type AboutSection = {
  id: string;
  number: string;
  title: string;
  photoSlot: string;
  photoCaption: string;
  /** Hint for aspect ratio when the real photo is loaded. */
  aspectHint?: '3/2' | '4/3' | '4/5';
  /** CSS object-position for the loaded photo. */
  objectPosition?: string;
  paragraphs: string[];
};

export const aboutStory = {
  eyebrow: 'about me',
  title: 'About Me',
  lede: 'A longer note on where I come from, how I found my way into software, and the mountains I still think of as home.',
  signoff: 'Feroz',
  sections: [
    {
      id: 'origin',
      number: '01',
      title: 'Where I Come From',
      photoSlot: 'where-i-come-from',
      photoCaption: 'The Kalash valleys, where I was born and raised.',
      aspectHint: '3/2',
      objectPosition: 'center 40%',
      paragraphs: [
        'I was born and raised in the Kalash valleys, surrounded by mountains, rivers, and a community with its own traditions, festivals, and way of life. I spent my early years and education there before moving to Chitral to continue my studies.',
        'Growing up in a place like that, you don\'t always realize how much it shapes you. When you are young, the mountains are simply where you live, the traditions are simply part of everyday life, and the people around you are simply your community. It is only after you leave that you begin to understand how much of yourself came from that place.',
        'I left home to continue my education and, eventually, to build a career. Since then, I have lived hundreds of kilometers away from my family and usually return only once or twice a year. With distance, I have come to appreciate the things I once took for granted, the festivals, gatherings, familiar faces, and the feeling of being somewhere that has always been home.',
        'I think a person\'s idea of home changes as they grow older. For me, it has become less about a particular house or village and more about the people, memories, traditions, and landscape that shaped me.',
        'And perhaps that is why, despite everything that has happened since I left, I still think of those mountains as somewhere I am trying to find my way back to.',
      ],
    },
    {
      id: 'camera',
      number: '02',
      title: 'A Camera in My Hands',
      photoSlot: 'camera-in-my-hands',
      photoCaption: 'Behind the lens — photography has been part of my life since 2014.',
      aspectHint: '3/2',
      objectPosition: 'center 35%',
      paragraphs: [
        'In 2014, I became interested in photography. I still do it from time to time and share some of my work on Instagram. Photography was one of the first things that taught me to pay attention to details and look at things differently. It is still something I enjoy, even though I don\'t have as much time for it now.',
        'That same year, I started working as a Non-Linear Video Editor at Leyenda Films in Islamabad, where I worked until 2017. It was my first professional experience, and it was also a time when I was still figuring out what I wanted to do with my career.',
      ],
    },
    {
      id: 'it',
      number: '03',
      title: 'Finding My Way Into IT',
      photoSlot: 'finding-it',
      photoCaption: 'University years — photo coming later',
      paragraphs: [
        'In spring 2017, I was admitted to Quaid-i-Azam University for a Master\'s in Information Technology. IT and software were completely new to me, and moving into the field was not always easy. I moved into the university hostel and completed my Master\'s in five semesters, finishing in mid-2019.',
        'During my time at the university, I became involved in the IT society. I ran for president and was elected by the students. Being part of the society and having the opportunity to lead it became an important part of my university experience. It also gave me the chance to work with people outside the classroom and take on responsibilities that were quite different from studying.',
        'When the COVID-19 pandemic started shortly after I graduated, I used the time to teach myself programming. I learned Python, PHP, and JavaScript, mostly without reliable internet access. I was learning by experimenting, building things, and trying to understand how software actually worked.',
        'This was when software development started becoming more than something I had studied at university. I began to see it as something I genuinely wanted to do.',
      ],
    },
    {
      id: 'work',
      number: '04',
      title: 'Making Software My Work',
      photoSlot: 'making-software',
      photoCaption: 'Software work — photo coming later',
      paragraphs: [
        'I started my professional software career in 2021 as a Backend Developer at Esols Technologies. I worked mainly with Node.js and hosted backend services on AWS Lightsail.',
        'In January 2023, I joined IT22 BV as a Backend Developer. This was where I moved from Node.js to Java and Quarkus. Over time, my work expanded beyond backend development, and I became involved with DevOps as well.',
        'Since 2025, I have been splitting my time roughly equally between Backend and DevOps. On the backend side, I continue to work mainly with Java and related technologies. On the DevOps side, I work with OpenShift/OKD, GitOps, ArgoCD, Tekton, containers, and related infrastructure.',
        'Over the years, I have worked with technologies including Java, Quarkus, Hibernate, Apache Camel, Kafka, MongoDB, MySQL, Node.js, Docker, Podman, Tekton, and ArgoCD.',
      ],
    },
    {
      id: 'curious',
      number: '05',
      title: 'What Keeps Me Curious',
      photoSlot: 'keeps-me-curious',
      photoCaption: 'Learning — photo coming later',
      paragraphs: [
        'I enjoy learning new things and figuring out how things work. I have completed several courses through the Red Hat partner program and continue to learn whenever I get the opportunity.',
        'Java remains one of my main areas of focus, and I particularly enjoy solving technical problems. More recently, I have been exploring artificial intelligence, agents, and agentic systems.',
        'I am still figuring out where that interest will take me, but I enjoy learning about the field and experimenting with new ideas.',
      ],
    },
    {
      id: 'home',
      number: '06',
      title: 'Finding My Way Back Home',
      photoSlot: 'way-back-home',
      photoCaption: 'The road back toward the mountains.',
      aspectHint: '4/5',
      objectPosition: 'center center',
      paragraphs: [
        'One of my long-term goals is to be able to work remotely from the mountains where I grew up and eventually spend the rest of my life there.',
        'Technology has taken me far from home, while the same technology is also what I hope will eventually allow me to return.',
        'For me, being able to work remotely from Kalash would mean more than simply working from home. It would allow me to be closer to my family and reconnect with the place, people, and traditions I have missed while living away, while still being able to do the work I enjoy.',
        'Maybe, in the end, that is the direction I am working toward: building things with technology while finding my way back home.',
      ],
    },
  ] satisfies AboutSection[],
} as const;
