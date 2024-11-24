export interface Model {
    label: string;
    value: string;
    breeds: string[];
}

export function parseBreed(breed: string): string {
    return breed.replace(/_/g, ' ');
}

const models: Model[] = [
    {
        label: '16 breeds model',
        value: '16_dogs',
        breeds: [
            'beagle',
            'boxer',
            'bulldog',
            'chihuahua',
            'corgi',
            'dachshund',
            'german_shepherd',
            'golden_retriever',
            'husky',
            'labrador',
            'pomeranian',
            'poodle',
            'pug',
            'rottweiler',
            'shiba_inu',
            'yorkshire_terrier'
        ]
    },
    {
        label: '8 breeds model',
        value: '8_dogs',
        breeds: [
            'beagle',
            'boxer',
            'golden_retriever',
            'husky',
            'poodle',
            'pug',
            'rottweiler',
            'yorkshire_terrier'
        ]    
    },
    {
        label: 'transfer learning model',
        value: 'pretrained',
        breeds: [
            'beagle',
            'boxer',
            'bulldog',
            'chihuahua',
            'corgi',
            'dachshund',
            'german_shepherd',
            'golden_retriever',
            'husky',
            'labrador',
            'pomeranian',
            'poodle',
            'pug',
            'rottweiler',
            'shiba_inu',
            'yorkshire_terrier'
        ]
    }
];

export default models;