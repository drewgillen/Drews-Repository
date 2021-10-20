import React from 'react';

import { Section, SectionText, SectionTitle } from '../../styles/GlobalComponents';
import Button from '../../styles/GlobalComponents/Button';
import { LeftSection } from './HeroStyles';

const Hero = () => (
  <Section row nopadding>
    <LeftSection>
      <SectionTitle main center>
      Welcome To <br />
      Drew Gillen's Portfolio
      </SectionTitle>
      <SectionText>
        I am the CEO of Abyiss
      </SectionText>
      <Button onClick ={() => window.location = 'https://abyiss.com'}>Learn More</Button>
    </LeftSection>
  </Section>
);

export default Hero;