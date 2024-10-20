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
        label: 'Dog',
        value: 'dog',
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
        label: 'Dog_v2',
        value: 'dog_v2',
        breeds: [
            'affenpinscher',
            'afghan_hound',
            'african_hunting_dog',
            'airedale',
            'american_staffordshire_terrier',
            'appenzeller',
            'australian_terrier'
        ]        
    }
];

export default models;